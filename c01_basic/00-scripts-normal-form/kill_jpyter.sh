ps -ef |grep cy.*jupyter |awk '{print $2}'|xargs kill -9
