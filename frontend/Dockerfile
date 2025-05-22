# Imagen base oficial de Nginx
FROM nginx:alpine

# Elimina la configuración por defecto de Nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copia tu propia configuración
COPY nginx.conf /etc/nginx/conf.d

# Copia los archivos estáticos al directorio donde Nginx los sirve
COPY . /usr/share/nginx/html

# Exponemos el puerto (Nginx escucha en el 80)
EXPOSE 80
