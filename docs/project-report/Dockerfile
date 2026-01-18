# Dockerfile for building PDF report with pandoc + LaTeX
FROM pandoc/extra:latest

# pandoc/extra includes:
# - pandoc
# - texlive with xelatex
# - common fonts
# - citeproc

RUN if command -v apt-get >/dev/null 2>&1; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            python3 \
            python3-pip \
            curl \
        && rm -rf /var/lib/apt/lists/*; \
    elif command -v apk >/dev/null 2>&1; then \
        apk add --no-cache python3 py3-pip curl; \
    else \
        echo "No supported package manager found (apt-get or apk)" >&2; \
        exit 1; \
    fi

RUN if command -v apt-get >/dev/null 2>&1; then \
        pip3 install --no-cache-dir pypandoc; \
    elif command -v apk >/dev/null 2>&1; then \
        apk add --no-cache py3-pypandoc || \
        pip3 install --no-cache-dir --break-system-packages pypandoc; \
    else \
        echo "No supported package manager found (apt-get or apk)" >&2; \
        exit 1; \
    fi

WORKDIR /workspace
