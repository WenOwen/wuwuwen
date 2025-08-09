git add .
git commit -m "xxx"
git push -u origin main   # 仅首次
# 后续
git push


# 清除Git凭据缓存
git config --global --unset credential.helper
git config --global --unset-all credential.helper

# 清除系统凭据存储
rm -f ~/.git-credentials

# 设置环境变量，禁用VSCode Git助手
export GIT_ASKPASS=""
export SSH_ASKPASS=""

#重新配置凭据存储
git config --global credential.helper store

#更新远程URL 
git remote set-url origin https://WenOwen@github.com/WenOwen/wuwuwen.git

#创建GitHub个人访问令牌
git push


# 查看当前远程配置
git remote -v

# 测试连接
git remote show origin