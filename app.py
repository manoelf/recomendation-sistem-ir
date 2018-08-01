from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from data import Articles
from movies_recomendatio import get_top5_by_array, show_top_3_neighborsshow_top, get_accuracy, get_already_watched
from wtforms import Form, StringField, TextAreaField, PasswordField, validators

app = Flask(__name__)

Articles = Articles()
Top5 = []
user = ''
closest_neighbors = []
rmse = -1
watched = []


@app.route('/', )
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/articles', methods=['GET', 'POST'])
def article():
    return render_template('articles.html', articles = Top5)

@app.route('/recomender', methods=['GET', 'POST'])
def recomender():
    form = UserSearchForm(request.form)
    if request.method == 'POST' and form.validate():
        user = form.user.data
        Top5 = get_top5_by_array(int(user))
        closest_neighbors = show_top_3_neighborsshow_top(user)
        rmse = get_accuracy()
        
        watched = get_already_watched(int(user))

        return render_template('articles.html', articles = Top5, closest_neighbors = closest_neighbors, user = user, rmse = rmse, watched = watched)
        
    return render_template('recomender.html', form=form)

class UserSearchForm(Form):
    user =  StringField('User', [validators.Length(min=1, max=3)])

if __name__ == '__main__':
    app.run(debug=True)
