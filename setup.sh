#!/bin/bash

echo "=== AutoRAG + JDK17 + Python requirements 설치 스크립트 ==="

# -----------------------
# 설치 전 확인
# -----------------------
echo "[1/4] Homebrew 존재 여부 확인..."
if ! command -v brew &> /dev/null
then
    echo "Homebrew가 설치되어 있지 않습니다. 먼저 brew를 설치해주세요."
    exit 1
fi

# -----------------------
# JDK17 설치
# -----------------------
echo "[2/4] JDK 17 설치 중..."
brew install openjdk@17

echo "[3/4] JDK 환경 변수 설정 중..."

# zsh 기준
SHELL_RC="$HOME/.zshrc"
JAVA_PATH="/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

if ! grep -q "openjdk@17" $SHELL_RC; then
    echo "export JAVA_HOME=\"$JAVA_PATH\"" >> $SHELL_RC
    echo "export PATH=\"\$JAVA_HOME/bin:\$PATH\"" >> $SHELL_RC
fi

source $SHELL_RC

echo "JAVA_HOME 설정됨: $JAVA_HOME"
java -version

# -----------------------
# Python requirements 설치
# -----------------------
echo "[4/4] requirements.txt 설치 시작..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=== 설치 완료! ==="

