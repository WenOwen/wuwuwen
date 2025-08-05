# 🐳 Docker跨网络部署方案

## 🎯 方案概述

**场景**: 原电脑和服务器不在同一局域网，无法直接HTTP传输
**解决方案**: Docker分离式部署 + 云端数据传输

```
原电脑 → 云端存储 → 服务器Docker容器
  5GB      压缩传输     自动下载解压
```

---

## 📋 核心架构

### 🔧 技术栈
- **容器化**: Docker + Docker Compose
- **数据分离**: 代码和数据完全分离
- **云端传输**: 百度网盘/阿里云OSS
- **自动部署**: 一键启动脚本

### 🎯 关键优势
- ✅ **轻量镜像**: 20MB代码镜像，快速下载
- ✅ **跨网络**: 通过云端中转数据
- ✅ **自动化**: 容器启动时自动下载数据
- ✅ **可重复**: 任意服务器都能快速部署

---

## 🚀 部署步骤

### 第一步：准备数据包（原电脑执行）

```bash
# 创建部署包
python baidu_uploader.py

# 输出示例：
# ✅ 部署包创建完成: wuwuquant_complete_20241228.tar.gz
# 📊 文件大小: 850 MB
# 📝 请上传到百度网盘/云存储
```

### 第二步：上传到云端

**方案A: 百度网盘（推荐个人用户）**
```bash
# 1. 手动上传 wuwuquant_complete_*.tar.gz 到百度网盘
# 2. 创建分享链接（永久有效）
# 3. 记录分享链接地址
```

**方案B: 阿里云OSS（推荐生产环境）**
```bash
# 一次性上传到OSS
ossutil cp wuwuquant_complete_*.tar.gz oss://your-bucket/
```

### 第三步：服务器Docker部署

```bash
# 1. 准备Docker环境
git clone your-repo
cd wuwuquant

# 2. 一键快速部署
python 快速部署.py

# 程序会自动：
# - 检查Docker环境
# - 配置数据源
# - 启动Docker服务
# - 自动下载数据

# 3. 访问服务
# Web界面: http://服务器IP:8501
# API文档: http://服务器IP:8000/docs
```

**手动配置方式：**
```bash
# 配置环境变量
echo "DATA_SOURCE_TYPE=baidu" > .env
echo "DATA_PACKAGE_URL=你的百度网盘链接" >> .env

# 启动Docker服务
docker-compose -f docker-compose.distributed.yml up -d
```

---

## 🔧 配置文件

### Docker Compose配置
```yaml
# docker-compose.distributed.yml
version: '3.8'
services:
  stock-prediction:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    volumes:
      - ./datas_em:/app/datas_em
      - ./models:/app/models
    environment:
      - DATA_SOURCE_TYPE=${DATA_SOURCE_TYPE}
      - DATA_PACKAGE_URL=${DATA_PACKAGE_URL}
    command: >
      sh -c "
        python data_downloader.py &&
        uvicorn prediction_service:app --host 0.0.0.0 --port 8000 &
        streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
      "
```

### 环境变量配置
```bash
# .env 文件
DATA_SOURCE_TYPE=baidu        # 或 aliyun
DATA_PACKAGE_URL=分享链接或OSS地址
AUTO_DOWNLOAD=true
```

---

## 📊 数据传输方案对比

| 方案 | 适用场景 | 操作复杂度 | 成本 | 速度 |
|------|----------|------------|------|------|
| **百度网盘** | 个人用户 | ⭐⭐ | 免费 | ⭐⭐⭐⭐ |
| **阿里云OSS** | 生产环境 | ⭐⭐⭐ | 低 | ⭐⭐⭐⭐⭐ |

---

## 🎯 完整操作流程

### 💻 原电脑操作（一次性）
```bash
# 1. 创建部署包
cd wuwuquant
python baidu_uploader.py

# 2. 上传到云端
# 百度网盘：手动上传，创建分享链接
# 阿里云OSS：ossutil cp package.tar.gz oss://bucket/
```

### 🖥️ 服务器操作（重复使用）
```bash
# 1. 克隆代码
git clone your-repo && cd wuwuquant

# 2. 配置数据源
echo "DATA_PACKAGE_URL=你的链接" > .env

# 3. 启动Docker
docker-compose -f docker-compose.distributed.yml up -d

# 4. 访问服务
# Web界面: http://服务器IP:8501
# API文档: http://服务器IP:8000/docs
```

---

## 🔍 故障排除

### 常见问题

**1. 数据下载失败**
```bash
# 检查日志
docker-compose logs stock-prediction

# 手动下载测试
docker exec -it container_name python data_downloader.py
```

**2. 端口占用**
```bash
# 修改端口映射
# 在 docker-compose.distributed.yml 中修改:
ports:
  - "8001:8000"  # API端口
  - "8502:8501"  # Web端口
```

**3. 内存不足**
```bash
# 限制容器内存使用
deploy:
  resources:
    limits:
      memory: 4G
```

---

## 📈 性能监控

```bash
# 查看容器状态
docker stats

# 查看服务日志
docker-compose logs -f stock-prediction

# 查看数据下载进度
docker exec -it container_name ls -la datas_em/
```

---

## 🎉 总结

这个方案完美解决了你的需求：
- ✅ **Docker分离式部署** - 代码和数据完全分离
- ✅ **跨网络传输** - 通过云端中转，无需同一局域网
- ✅ **自动化程度高** - 一键部署，自动下载数据
- ✅ **可重复使用** - 任意服务器都能快速启动

现在你可以在任何有Docker的服务器上，用几条命令完成整个系统的部署！🚀