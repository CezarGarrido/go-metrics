#!/bin/bash

# Script para clonar 20 projetos Go com git clone --depth 1
# Aplicações de usuário final: e-commerce, ERP, APIs, etc.

echo "Clonando 20 projetos Go para análise..."
echo "========================================"

# Criar diretório para os projetos
mkdir -p go_user_apps
cd go_user_apps

# Lista de projetos para clonar
git clone --depth 1 https://github.com/gothinkster/golang-gin-realworld-example-app
git clone --depth 1 https://github.com/bxcodec/go-clean-arch
git clone --depth 1 https://github.com/mattermost/mattermost
git clone --depth 1 https://github.com/gohugoio/hugo
git clone --depth 1 https://github.com/answerdev/answer
git clone --depth 1 https://github.com/go-sonic/sonic
git clone --depth 1 https://github.com/plausible/analytics
git clone --depth 1 https://github.com/bytebase/bytebase
git clone --depth 1 https://github.com/actualbudget/actual
git clone --depth 1 https://github.com/amir20/dozzle
git clone --depth 1 https://github.com/iv-org/invidious
git clone --depth 1 https://github.com/zedeus/nitter
git clone --depth 1 https://github.com/ponzu-cms/ponzu
git clone --depth 1 https://github.com/micro/micro
git clone --depth 1 https://github.com/urfave/cli
git clone --depth 1 https://github.com/spf13/cobra
git clone --depth 1 https://github.com/gin-gonic/gin
git clone --depth 1 https://github.com/labstack/echo
git clone --depth 1 https://github.com/go-gorm/gorm
git clone --depth 1 https://github.com/valyala/fasthttp

echo "========================================"
echo "Clone concluído! Projetos salvos em: $(pwd)"
