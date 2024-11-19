import argparse
import json
from os import listdir
from os.path import join

from dbqrqa.dataset import TableSplit
from dbqrqa.evaluation import evaluate

DEFAULT_DATA_PATH = 'data'
DEFAULT_GPT_MODEL = 'gpt-4o'
DEFAULT_RETRY = 3


def run(
    data_path: str, 
    branch: str, 
    stage: str,
    since: str, 
    retry: int):

    answer_path = join(data_path, 'answers', branch)
    users = listdir(answer_path)

    with open(join(data_path, 'keys', 'openai.txt')) as reader:
        openai_key = reader.read().strip()

    for user in users:
        score_file = join(data_path, 'scores', branch, f'{user}.json')
        
        with open(score_file) as reader:
            records = json.load(reader)

        scores = []

        for record in records:
            if record['stage'].lower() == stage and \
                record['submitted'][:10] >= since:
                scores.append(record)

        rank = sorted(scores, key=lambda x: x['graderScore'], reverse=True)

        if len(rank) == 0:
            continue

        top = rank[0]

        stage_path = join(data_path, 'gold', 'branches', branch, 'dataset')

        data = TableSplit(stage, stage_path)

        timestamp = top['timestamp']
        answer_file = join(data_path, 'answers', branch, user, f'{timestamp}.json')

        with open(answer_file) as reader:
            data.answers = json.load(reader)

        accuracy, _ = data.evaluate()
        print('Evaluating ' + user + ', grader score = %.2f' % accuracy)

        output_path = join(data_path, 'gpt', branch, user, stage, timestamp)

        accuracy, scores = data.evaluate(
            evaluator='gpt-binary',
            openai_key=openai_key,
            backup_path=join(output_path, 'backup'),
            retry=retry,
            is_notebook=False)

        with open(join(output_path, 'accuracy.txt'), 'w') as writer:
            writer.write('%f' % accuracy)

        with open(join(output_path, 'scores.json'), 'w') as writer:
            json.dump(scores, writer, indent=2)


def main(args: argparse.Namespace):
    run(args.data, args.branch, args.stage, args.since, args.retry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('since', type=str)
    parser.add_argument('--branch', type=str, default='master')
    parser.add_argument('--stage', type=str, default='test')
    parser.add_argument('--model', type=str, default=DEFAULT_GPT_MODEL)
    parser.add_argument('--retry', type=int, default=DEFAULT_RETRY)
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    main(args)
