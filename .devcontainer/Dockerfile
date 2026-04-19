FROM mcr.microsoft.com/devcontainers/python:3.11

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        biber \
        chktex \
        latexmk \
        shfmt \
        texlive-bibtex-extra \
        texlive-extra-utils \
        texlive-fonts-recommended \
        texlive-lang-english \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-luatex \
        texlive-pictures \
        texlive-science \
        texlive-xetex \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
