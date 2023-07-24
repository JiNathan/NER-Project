from flask import Flask, jsonify

app = Flask(__name__)

# Sample data for wordsToHighlight
wordsToHighlight = ['example', 'highlight']

@app.route('/get_words_to_highlight', methods=['GET'])
def get_words_to_highlight():
    return jsonify(wordsToHighlight)

if __name__ == '__main__':
    app.run()