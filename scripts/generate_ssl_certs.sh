#!/bin/bash
# Generate self-signed SSL certificates for staging environment
# These are for testing only - use proper certificates in production

set -e

CERT_DIR="nginx/ssl"
DOMAIN="staging.musicgen-ai.local"

# Create SSL directory if it doesn't exist
mkdir -p "$CERT_DIR"

echo "Generating self-signed SSL certificates for staging..."

# Generate private key
openssl genrsa -out "$CERT_DIR/staging.key" 4096

# Generate certificate signing request
cat > "$CERT_DIR/staging.conf" << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = California
L = San Francisco
O = Music Gen AI
OU = Staging Environment
CN = $DOMAIN

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = localhost
DNS.3 = staging-api.musicgen-ai.local
DNS.4 = musicgen-api-1
DNS.5 = musicgen-api-2
DNS.6 = nginx-staging
IP.1 = 127.0.0.1
IP.2 = 172.20.0.0/16
EOF

# Generate certificate
openssl req -new -x509 -key "$CERT_DIR/staging.key" \
    -out "$CERT_DIR/staging.crt" \
    -config "$CERT_DIR/staging.conf" \
    -extensions v3_req \
    -days 365

# Set appropriate permissions
chmod 600 "$CERT_DIR/staging.key"
chmod 644 "$CERT_DIR/staging.crt"

echo "SSL certificates generated successfully:"
echo "  Certificate: $CERT_DIR/staging.crt"
echo "  Private Key: $CERT_DIR/staging.key"
echo ""
echo "To trust the certificate locally (macOS):"
echo "  sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $CERT_DIR/staging.crt"
echo ""
echo "To trust the certificate locally (Linux):"
echo "  sudo cp $CERT_DIR/staging.crt /usr/local/share/ca-certificates/"
echo "  sudo update-ca-certificates"

# Verify certificate
echo ""
echo "Certificate details:"
openssl x509 -in "$CERT_DIR/staging.crt" -text -noout | grep -A 5 "Subject:"
openssl x509 -in "$CERT_DIR/staging.crt" -text -noout | grep -A 5 "Subject Alternative Name"