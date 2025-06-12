1.
만약 tensorflow 설치에 문제가 있다면 python 버전을 낮춰야 함
python은 3.10 이하의 버전으로 해야 tensorflow 작동이 가능함 

2.
웹페이지 실행은 main.py를 실행시킨 후 http://localhost:8000/로 접속

3.
mp3나 wav가 아닌 오디오 파일 업로드하면 에러 남

4.
git pull 하고 나서 프로젝트 경로에 mkdir static 해야지 안 하면 에러 발생
fastapi 특성 상 static이라는 파일을 마운트하는데,
해당 gitignore로 인해 누락됨
