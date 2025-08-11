# Git实用命令参考指南

> 基于实际使用需求整理的Git命令集合

## 目录
- [用户配置管理](#用户配置管理)
- [远程仓库配置](#远程仓库配置)
- [查看项目状态](#查看项目状态)
- [查看提交历史](#查看提交历史)
- [拉取和推送操作](#拉取和推送操作)
- [分支管理](#分支管理)
- [凭据管理](#凭据管理)
- [网络问题解决](#网络问题解决)
- [Git分页器使用](#git分页器使用)
- [常见问题解决](#常见问题解决)

---

## 用户配置管理

### 查看当前用户信息
```bash
# 查看用户名
git config user.name

# 查看邮箱
git config user.email

# 查看所有配置
git config --list | grep user
```

### 设置用户信息
```bash
# 全局设置（影响所有仓库）
git config --global user.name "您的用户名"
git config --global user.email "您的邮箱"

# 仅当前项目设置
git config user.name "您的用户名"
git config user.email "您的邮箱"

# 推荐：使用GitHub noreply邮箱保护隐私
git config --global user.email "用户ID+用户名@users.noreply.github.com"
```

### 查看所有配置
```bash
# 查看全局配置
git config --global --list

# 查看当前仓库配置
git config --list
```

---

## 远程仓库配置

### 查看远程仓库
```bash
# 查看远程仓库列表
git remote -v

# 查看远程仓库详细信息
git remote show origin
```

### 设置和修改远程仓库
```bash
# 设置远程仓库URL
git remote set-url origin https://用户名@github.com/用户名/仓库名.git

# 添加新的远程仓库
git remote add origin https://github.com/用户名/仓库名.git

# 删除不需要的远程仓库
git remote remove upstream
```

### 验证远程连接
```bash
# 测试远程连接
git ls-remote origin

# 查看远程分支
git ls-remote origin
```

---

## 查看项目状态

### 基本状态检查
```bash
# 查看工作区状态
git status

# 查看当前分支
git branch

# 查看所有分支（包括远程）
git branch -a

# 查看分支详细信息
git branch -vv
```

### 比较差异
```bash
# 查看工作区与暂存区差异
git diff

# 查看暂存区与仓库差异
git diff --cached

# 查看本地与远程差异
git diff main origin/main
```

---

## 查看提交历史

### 基础历史查看
```bash
# 查看完整提交历史
git log

# 查看简洁的一行历史
git log --oneline

# 查看最近N次提交
git log --oneline -10

# 退出分页器
q
```

### 图形化历史
```bash
# 查看分支图形化历史
git log --oneline --graph

# 查看所有分支的图形化历史
git log --oneline --graph --all

# 限制显示数量
git log --oneline --graph -10
```

### 远程历史查看
```bash
# 先获取远程信息
git fetch origin

# 查看远程分支历史
git log origin/main --oneline

# 查看本地与远程的差异
git log main..origin/main --oneline
git log origin/main..main --oneline
```

### 禁用分页器的方法
```bash
# 临时禁用分页器
git --no-pager log --oneline

# 全局禁用分页器
git config --global core.pager ""

# 设置友好的分页器
git config --global core.pager "less -FRX"
```

---

## 拉取和推送操作

### 安全的拉取流程
```bash
# 1. 获取远程最新信息（不合并）
git fetch origin

# 2. 查看将要合并的内容
git log main..origin/main --oneline

# 3. 确认无问题后合并
git merge origin/main

# 或者一步完成（fetch + merge）
git pull origin main
```

### 推送操作
```bash
# 推送到远程仓库
git push origin main

# 首次推送设置上游分支
git push -u origin main

# 强制推送（慎用）
git push --force origin main
```

### 处理冲突
```bash
# 如果拉取时有冲突
git pull origin main

# 查看冲突文件
git status

# 编辑解决冲突后
git add 冲突文件名
git commit -m "解决合并冲突"
```

---

## 分支管理

### 查看分支
```bash
# 查看本地分支
git branch

# 查看所有分支
git branch -a

# 查看远程分支
git branch -r
```

### 分支操作
```bash
# 创建并切换分支
git checkout -b 新分支名

# 切换分支
git checkout 分支名

# 删除分支
git branch -d 分支名

# 强制删除分支
git branch -D 分支名
```

---

## 凭据管理

### 查看凭据配置
```bash
# 查看凭据助手设置
git config credential.helper

# 查看是否有存储的凭据
ls -la ~/.git-credentials
```

### 配置凭据存储
```bash
# 设置凭据存储
git config --global credential.helper store

# Windows系统推荐使用
git config --global credential.helper manager-core

# 清除存储的凭据
git config --global --unset credential.helper
rm -f ~/.git-credentials
```

### GitHub个人访问令牌
```bash
# 生成令牌后，推送时输入：
# 用户名：您的GitHub用户名
# 密码：粘贴个人访问令牌（不是GitHub登录密码）

# 令牌获取地址：
# https://github.com/settings/tokens
```

---

## 网络问题解决

### 常见网络配置
```bash
# 增加超时时间
git config --global http.timeout 60
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999

# 清除代理配置
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### SSL证书问题
```bash
# Windows系统SSL配置
git config --global http.sslBackend schannel
git config --global http.sslCAInfo ""

# 临时禁用SSL验证（不推荐）
git config --global http.sslVerify false
```

### 网络诊断
```bash
# 测试GitHub连接
ping github.com

# 测试HTTPS连接
curl -I https://github.com/用户名/仓库名.git

# 测试Git连接
git ls-remote origin
```

---

## Git分页器使用

### 分页器操作
```bash
# 导航操作
空格键 或 Page Down     # 向下翻页
回车键                  # 向下滚动一行
b 或 Page Up           # 向上翻页
上/下箭头              # 逐行滚动
g                      # 跳到开头
G                      # 跳到结尾

# 退出操作
q                      # 退出查看界面
Ctrl+C                 # 强制退出
```

### 分页器配置
```bash
# 设置更友好的分页器
git config --global core.pager "less -FRX"

# 禁用特定命令的分页器
git config --global pager.log false

# 临时禁用分页器
git --no-pager log --oneline
```

---

## 常见问题解决

### 问题1：认证失败
```bash
# 解决方案：
git remote set-url origin https://用户名@github.com/用户名/仓库名.git
git config --global credential.helper store
# 推送时输入个人访问令牌
```

### 问题2：本地与远程不同步
```bash
# 诊断：
git fetch origin
git log main..origin/main --oneline

# 解决：
git pull origin main
# 或强制同步（慎用）：
git reset --hard origin/main
```

### 问题3：网络连接失败
```bash
# 解决步骤：
git config --global http.timeout 60
git config --global --unset http.proxy
git remote set-url origin https://用户名@github.com/用户名/仓库名.git
git fetch origin
```

### 问题4：误删除upstream
```bash
# 如果是自己的项目，直接删除：
git remote remove upstream

# 验证：
git remote -v
git branch -a
```

---

## 实用技巧

### 快速状态检查
```bash
# 一键检查项目状态
echo "=== 用户信息 ==="
git config user.name && git config user.email

echo "=== 远程配置 ==="
git remote -v

echo "=== 分支状态 ==="
git branch -vv

echo "=== 工作区状态 ==="
git status
```

### 同步检查流程
```bash
# 完整的同步检查流程
git fetch origin
git status
git log main..origin/main --oneline
# 如果有更新：
git pull origin main
```

### 推荐的全局配置
```bash
# 设置友好的Git环境
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.pager "less -FRX"
git config --global credential.helper store
```

---

## 紧急情况处理

### 如果搞乱了，重新开始
```bash
# 1. 备份当前工作
cp -r . ../项目备份

# 2. 重新克隆
cd ..
git clone https://github.com/用户名/仓库名.git 新目录

# 3. 恢复工作文件
cp ../项目备份/重要文件 新目录/
```

### 撤销操作
```bash
# 撤销最后一次提交（保留文件修改）
git reset --soft HEAD~1

# 撤销工作区修改
git checkout -- 文件名

# 撤销暂存区修改
git reset HEAD 文件名
```

---

**提示**：建议收藏这个文档，遇到Git问题时快速查找相应的解决方案！