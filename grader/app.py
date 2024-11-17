import datetime
import json
import math
import os
from os import listdir
from os.path import isdir, isfile, join
from pathlib import Path

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from dbqrqa.evaluation import evaluate_heuristic

DEFAULT_DATA_PATH = '../data'
N_SAMPLES = {'practice': 5, 'training': 20, 'test': 15}
SUBMISSION_LITMIT = 100
PAGE_SIZE = 5

app = Flask(__name__)
CORS(app)


def _get_user_info(token: str) -> dict:
    users_path = join(DEFAULT_DATA_PATH, 'users')
    
    for username in listdir(users_path):
        user_path = join(users_path, username)
        
        if not isdir(user_path):
            continue
        
        account_path = join(users_path, username, 'account.json')
        
        with open(account_path) as reader:
            account = json.load(reader)
        
        if account['token'] == token:
            return {'username': username, **account}
    
    return None


def _get_stage(answers: dict) -> str:
    for stage in ('practice', 'training', 'test'):
        if len(answers) == N_SAMPLES[stage]:
            return stage

    return None


def _get_limit(username):
    score_file = join(DEFAULT_DATA_PATH, 'users', username, 'scores.json')
    
    if isfile(score_file):
        with open(score_file) as reader:
            scores = json.load(reader)
            
        now = datetime.datetime.now()
        dformat = '%Y-%m-%d'
        ref = now.strftime(dformat)
        remaining = SUBMISSION_LITMIT
        
        for score in scores:
            if score['timestamp'].startswith(ref):
                remaining -= 1
        
        return remaining
    
    else:
        return SUBMISSION_LITMIT


def _evaulate(answers: dict, labels: dict) -> float:
    scores = []
    
    for conv, samples in labels.items():
        if conv not in answers:
            return None
        
        for qid, label in samples.items():
            if qid not in answers[conv]:
                return None
            
            answer = answers[conv][qid]
            score = evaluate_heuristic(answer, label)
            scores.append(score)
    
    return sum(scores) / len(scores)


@app.route('/')
def top():
    return 'DBQR-QA Grader Version 0.1.0: Running'


@app.route('/')
def index():
    return top()


@app.route('/dbqr-qa/status')
def status():
    return jsonify({'status': 'ok'})
    
    
@app.route('/dbqr-qa/leaderboard')
def leaderboard():
    users_path = join(DEFAULT_DATA_PATH, 'users')
    data = {}
    
    for stage in ('practice', 'training', 'test'):
        users = []
        
        for user in listdir(users_path):
            user_path = join(users_path, user)
            account_file = join(user_path, 'account.json')
            scores_file = join(user_path, 'scores.json')
            
            if not isfile(scores_file):
                continue
            
            with open(account_file) as reader:
                account = json.load(reader)
            
            with open(scores_file) as reader:
                history = json.load(reader)
            
            scores = []
            
            for record in history:
                if record['stage'].lower() == stage:
                    scores.append(record)
                    
            if (len(scores)) == 0:
                continue
                    
            best = sorted(scores, key=lambda x: x['graderScore'], reverse=True)[0]
            
            users.append({
                'name': account['display'],
                'entries': len(scores),
                'graderScore': best['graderScore'],
                'gptScore': best['gptScore'],
                'humanScore': best['humanScore'],
                'last': scores[-1]['submitted']})
        
        users = sorted(users, key=lambda x: x['graderScore'], reverse=True)
        data[stage] = users
    
    return jsonify({
        'status': 'ok',
        'scores': data})
    
    
@app.route('/dbqr-qa/activate')
def activate():
    token = request.args.get('token', None)
    account = _get_user_info(token)
    
    if account is None:
        return jsonify({
            'status': 'error',
            'error': 'invalid_token',
            'message': 'Invalid token.'})
    
    return jsonify({
        'status': 'ok',
        'account': account})
    

@app.route('/dbqr-qa/history')
def history():
    token = request.args.get('token', None)
    page = request.args.get('page', 1)
    
    if isinstance(page, str):
        if page.isdigit():
            page = int(page)
        
        else:
            page = 1
    
    account = _get_user_info(token)
    
    if account is None:
        return jsonify({
            'status': 'error',
            'error': 'invalid_token',
            'message': 'Invalid token.'})
        
    username = account['username']
        
    score_file = join(DEFAULT_DATA_PATH, 'users', username, 'scores.json')
    
    if isfile(score_file):
        with open(score_file) as reader:
            history = json.load(reader)
            
    else:
        return jsonify({
            'status': 'ok',
            'history': [],
            'records': 0,
            'page': 1,
            'pageCount': 1,
            'pageSize': PAGE_SIZE,
            'remaining': _get_limit(username)})
    
    history = list(reversed(history))
    
    page_count = math.ceil(len(history) / PAGE_SIZE)
    page = min(max(page, 1), page_count) - 1
    
    return jsonify({
        'status': 'ok',
        'history': history[page * PAGE_SIZE:(page + 1) * PAGE_SIZE],
        'records': len(history),
        'page': page + 1,
        'pageCount': page_count,
        'pageSize': PAGE_SIZE,
        'remaining': _get_limit(username)})


@app.route('/dbqr-qa/submit', methods=['POST',])
def submit():
    token = request.form.get('token', None)
    
    account = _get_user_info(token)
    
    if account is None:
        return jsonify({
            'status': 'error',
            'error': 'invalid_token',
            'message': 'Invalid token.'})
        
    username = account['username']
    
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'error': 'no_file',
            'message': 'No file attached.'})
    
    file = request.files['file']
    
    if not file.filename.endswith('.json'):
        return jsonify({
            'status': 'error',
            'error': 'invalid_file_type',
            'message': 'Invalid file type.'})
        
    if _get_limit(username) == 0:
        return jsonify({
            'status': 'error',
            'error': 'submission_limit_exceeded',
            'message': 'Daily submission limit exceeded.'})
    
    now = datetime.datetime.now()
    tformat = '%Y-%m-%d_%H-%M-%S_%f'
    sformat = '%Y-%m-%d %H:%M:%S_%f'
    timestamp = now.strftime(tformat)
    submitted = now.strftime(sformat)
    
    path = join(DEFAULT_DATA_PATH, 'users', username, 'answers')
    Path(path).mkdir(exist_ok=True, parents=True)
    stored_file = join(path, f'{timestamp}.json')
    
    file.save(stored_file)
    
    try:
        with open(stored_file) as reader:
            answers = json.load(reader)
    
    except:
        os.remove(stored_file)
        
        return jsonify({
            'status': 'error',
            'error': 'incorrect_file',
            'message': 'Unable to read the file. Make sure it is in a correct JSON format.'})
        
    stage = _get_stage(answers)
        
    if stage is None:
        os.remove(stored_file)
        
        return jsonify({
            'status': 'error',
            'error': 'stage_not_found',
            'message': 'The answers do not match any of stages (practice/training/test).'})
    
    label_file = join(DEFAULT_DATA_PATH, 'gold', 'compiled', f'{stage}.json')
    
    with open(label_file) as reader:
        labels = json.load(reader)
    
    try:
        score = _evaulate(answers, labels)
    
    except:
        os.remove(stored_file)
        
        return jsonify({
            'status': 'error',
            'error': 'evaluation_failed',
            'message': 'Unknown error occurred during the evaluation.'})
    
    if score is None:
        os.remove(stored_file)
        
        return jsonify({
            'status': 'error',
            'error': 'missing_answers',
            'message': 'Missing answers for at least one of the conversations.'})
        
    score_file = join(DEFAULT_DATA_PATH, 'users', username, 'scores.json')
    
    if isfile(score_file):
        with open(score_file) as reader:
            scores = json.load(reader)
            
    else:
        scores = []
        
    scores.append({
        'entry': len(scores) + 1,
        'submitted': submitted[:19],
        'timestamp': timestamp,
        'stage': stage.capitalize(),
        'status': 'Success',
        'graderScore': score,
        'gptScore': '-',
        'humanScore': '-'})
    
    with open(score_file, 'w') as writer:
        json.dump(scores, writer, indent=2)
    
    return jsonify({'status': 'ok'})


@app.route('/dbqr-qa/username', methods=['POST',])
def username():
    token = request.form.get('token', None)
    name = request.form.get('name', None)
    
    account = _get_user_info(token)
    
    if account is None:
        return jsonify({
            'status': 'error',
            'error': 'invalid_token',
            'message': 'Invalid token.'})
        
    username = account['username']
    account['display'] = name
    user_path = join(DEFAULT_DATA_PATH, 'users', username)
    account_path = join(user_path, 'account.json')
        
    with open(account_path, 'w') as writer:
        json.dump(account, writer, indent=2)
    
    return {
        'status': 'ok',
        'name': account['display']}


@app.route('/dbqr-qa/limit')
def limit():
    token = request.form.get('token', None)
    
    account = _get_user_info(token)
    
    if account is None:
        return jsonify({
            'status': 'error',
            'error': 'invalid_token',
            'message': 'Invalid token.'})
        
    username = account['username']
    remaining = _get_limit(username)
        
    return {
        'status': 'ok',
        'remaining': remaining}


@app.route('/dbqr-qa/download')
def download():
    token = request.args.get('token', None)
    timestamp = request.args.get('timestamp', None)
    
    account = _get_user_info(token)
    
    if account is None:
        return 'Invalid token.'

    username = account['username']
    file = join(DEFAULT_DATA_PATH, 'users', username, 'answers', f'{timestamp}.json')
    
    if isfile(file):
        return send_file(file, as_attachment=True)
    
    else:
        return f'Answer not found for {timestamp}'
