from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField

class ActualAdmitsInputForm(FlaskForm):
    number = StringField('Number', render_kw={"name": "number_of_unis"})
    submit = SubmitField('Submit')