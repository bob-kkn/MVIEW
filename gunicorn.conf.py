# gunicorn.conf.py
def on_starting(server):
    # 마스터 프로세스에서 1회 실행
    import app
    app.warmup_indexes()