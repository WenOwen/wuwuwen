# Git 远程拉取命令指南

## 基本拉取命令

### 1. git pull - 拉取并合并
```bash
# 拉取当前分支的远程更新并自动合并
git pull

# 拉取指定远程仓库的指定分支
git pull origin main

# 拉取时使用rebase而不是merge
git pull --rebase

# 拉取时强制覆盖本地更改（危险操作）
git pull --force
```

### 2. git fetch - 仅拉取不合并
```bash
# 拉取所有远程分支的更新，但不合并
git fetch

# 拉取指定远程仓库的更新
git fetch origin

# 拉取指定分支的更新
git fetch origin main

# 拉取所有远程仓库的更新
git fetch --all

# 拉取时删除远程已删除的分支引用
git fetch --prune
```

## 高级拉取操作

### 3. 处理冲突的拉取
```bash
# 当有未提交的更改时，先暂存再拉取
git stash
git pull
git stash pop

# 使用rebase策略拉取以保持线性历史
git pull --rebase origin main

# 拉取时自动解决简单冲突
git pull --strategy-option=ours
git pull --strategy-option=theirs
```

### 4. 检查远程状态
```bash
# 查看远程仓库信息
git remote -v

# 查看远程分支
git branch -r

# 查看所有分支（本地+远程）
git branch -a

# 查看远程仓库详细信息
git remote show origin
```

### 5. 特殊拉取场景
```bash
# 拉取特定数量的提交历史
git pull --depth=10

# 拉取时只获取指定文件
git pull origin main -- 文件名

# 拉取并重置到远程状态（丢弃本地更改）
git fetch origin
git reset --hard origin/main

# 拉取tags
git fetch --tags

# 拉取时显示详细信息
git pull --verbose
```

## 配置相关

### 6. 配置默认拉取行为
```bash
# 设置pull默认使用rebase
git config --global pull.rebase true

# 设置pull默认使用merge（默认行为）
git config --global pull.rebase false

# 设置pull只有在快进时才执行
git config --global pull.ff only

# 查看当前配置
git config --list | grep pull
```

### 7. 设置上游分支
```bash
# 设置当前分支的上游分支
git branch --set-upstream-to=origin/main

# 推送时设置上游分支
git push -u origin main

# 查看分支的上游信息
git branch -vv
```

## 实用技巧

### 8. 安全拉取流程
```bash
# 1. 查看当前状态
git status

# 2. 暂存未完成的工作
git stash

# 3. 拉取远程更新
git pull origin main

# 4. 恢复之前的工作
git stash pop

# 5. 处理可能的冲突
# （如果有冲突，解决后提交）
```

### 9. 批量拉取多个分支
```bash
# 拉取所有远程分支
git fetch --all

# 更新所有本地分支到最新
for branch in $(git branch | sed 's/*//' | sed 's/ //'); do
    git checkout $branch
    git pull origin $branch
done
```

### 10. 撤销拉取操作
```bash
# 查看拉取前的提交ID
git reflog

# 重置到拉取前的状态
git reset --hard HEAD@{1}

# 或者撤销最近的merge
git reset --hard HEAD~1
```

## 常见问题解决

### 拉取失败的常见原因及解决方案

1. **网络问题**
```bash
# 设置代理
git config --global http.proxy http://proxy.server.com:port
git config --global https.proxy https://proxy.server.com:port

# 取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

2. **认证问题**
```bash
# 重新配置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 清除保存的密码
git config --global --unset credential.helper
```

3. **SSL证书问题**
```bash
# 临时跳过SSL验证（不推荐在生产环境使用）
git config --global http.sslVerify false

# 恢复SSL验证
git config --global http.sslVerify true
```

## 最佳实践

1. **拉取前先检查状态**: 始终在拉取前运行 `git status`
2. **定期拉取**: 养成定期拉取远程更新的习惯
3. **使用fetch+merge**: 在重要项目中考虑使用 `git fetch` 然后手动 `git merge`
4. **备份重要工作**: 在拉取前备份重要的未提交更改
5. **理解rebase vs merge**: 根据团队协作策略选择合适的拉取方式

---

*此文档涵盖了git远程拉取的常用命令和场景，建议根据具体项目需求选择合适的命令。*