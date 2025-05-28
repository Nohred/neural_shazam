FROM nginx:alpine

# Eliminar la config por defecto de nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copiar nginx.conf desde esta carpeta (frontend/)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar el contenido del frontend (todos los archivos estáticos)
COPY . /usr/share/nginx/html

EXPOSE 80
