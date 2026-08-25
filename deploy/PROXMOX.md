# Self-hosted deployment (Proxmox LXC)

This is the path for running THE OCTAGON on your own homelab instead of
Railway — use it if/when you want a self-hosted option, not as a second
copy of the same production deployment. See the root README's
"Where this actually runs" section for how this relates to Railway.

## 1. Prepare the LXC

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl git docker.io docker-compose-v2 python3-pip
sudo usermod -aG docker $USER
newgrp docker

mkdir -p ~/octagon
cd ~/octagon
# clone the repo here, or rsync it over
```

In the LXC config on the Proxmox host (`/etc/pve/lxc/<CTID>.conf`), enable
nesting so Docker-in-LXC works:

```
features: nesting=1,keyctl=1
```

## 2. Run it

```bash
cd deploy
docker compose -f docker-compose.prod.yml up -d
```

## 3. Run it 24/7 with systemd

Copy `deploy/octagon.service` to `/etc/systemd/system/octagon.service`,
fill in your real username and path, then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now octagon.service
journalctl -u octagon.service -f
```

## 4. Expose it publicly with Cloudflare Tunnel (no open ports)

```bash
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

Create a tunnel in the Cloudflare Zero Trust dashboard (Networks → Tunnels),
install it with the token it gives you:

```bash
sudo cloudflared service install <TUNNEL_TOKEN>
sudo systemctl enable --now cloudflared
```

Then add a Published Application Route: subdomain of your choice, type
HTTP, target `http://localhost:8080`. Cloudflare creates the DNS record
automatically and terminates TLS for you.

## Common problems

| Problem | Fix |
|---|---|
| Docker permission denied | `newgrp docker` or log out/in |
| Nested Docker fails | confirm `nesting=1` in the LXC config, reboot the container |
| 502 / connection refused through the tunnel | confirm the container listens on `0.0.0.0:8080`, not `127.0.0.1` |
| Port conflict | change the host-side port in `docker-compose.prod.yml` |
