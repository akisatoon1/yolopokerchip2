#!/usr/bin/env bash

# for git commit
git config --global user.email "akisatoon@gmail.com"
git config --global user.name "Akisatoon"

# for git push
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
