FROM docker.io/alpine:3.20

RUN apk add --no-cache curl jq

COPY init.sh /init.sh

# Normalisasi CRLF: file yang dibuat di Windows akan merusak shebang.
RUN sed -i 's/\r$//' /init.sh

ENTRYPOINT ["/bin/sh", "/init.sh"]
