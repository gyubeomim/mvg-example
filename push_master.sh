git add . -A

OS=$(uname)
if [[ "$OS" == "Darwin" ]]; then
	if [ -z "$1" ]
	then
		git commit -m "$(date +%Y%m%d) : OSX" 
	else
		git commit -m "$1" 
	fi
else
	if [ -z "$1" ]
	then
		git commit -m "$(date +%Y%m%d) : $(lsb_release -sd)" 
	else
		git commit -m "$1" 
	fi
fi
git push origin master
