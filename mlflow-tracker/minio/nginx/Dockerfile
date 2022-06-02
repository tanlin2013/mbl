FROM nginx:1.17.6

# Remove default Nginx config
RUN rm /etc/nginx/nginx.conf

# Copy the modified Nginx conf
COPY nginx.conf /etc/nginx

# Copy proxy config
COPY mlflow.conf /etc/nginx/sites-enabled/

# TLS Certificate with Let's encrypt
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:certbot/certbot

RUN apt install -y python-certbot-nginx && \
    certbot --nginx --rsa-key-size 4096 --no-redirect --staple-ocsp -d do-doublets.phys.nthu.edu.tw
