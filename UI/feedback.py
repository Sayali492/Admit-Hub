from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, StringField, SubmitField, TextAreaField

class FeedbackForm(FlaskForm):
    university1 = StringField('University1', render_kw={"name": "university1"})
    university2 = StringField('University2', render_kw={"name": "university2"})
    university3 = StringField('University3', render_kw={"name": "university3"})
    submit = SubmitField('Submit feedback')  # Submit button for form submission