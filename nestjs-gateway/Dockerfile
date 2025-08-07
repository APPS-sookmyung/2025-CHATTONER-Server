# nestjs-gateway/Dockerfile

FROM node:18

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 설치
COPY package*.json ./
RUN npm install

# 소스 복사 및 빌드
COPY . .
RUN npm run build

# 앱 실행
CMD ["npm", "run", "start:prod"]
