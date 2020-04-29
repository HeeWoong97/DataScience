#!"C:\Users\User\Desktop\새 폴더\공부\프로그래밍\DataScience\실습\2) 기본적인 Machine Learning 모형\1) Naive_Bayes_실습\venv\Scripts\python.exe" -x
# EASY-INSTALL-ENTRY-SCRIPT: 'pip==19.0.3','console_scripts','pip3.8'
__requires__ = 'pip==19.0.3'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pip==19.0.3', 'console_scripts', 'pip3.8')()
    )
