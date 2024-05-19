from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, StringField, SubmitField, TextAreaField

class InputCourseForm(FlaskForm):
    gpa = FloatField('GPA', render_kw={"name": "gpa"})  # Assuming GPA can be a float
    gre_awa = FloatField('GRE_Awa', render_kw={"name": "gre_awa"})
    gre_verbal = IntegerField('GRE_Verbal', render_kw={"name": "gre_verbal"})
    gre_quant = IntegerField('GRE_Quant', render_kw={"name": "gre_quant"})
    toefl = IntegerField('Toefl', render_kw={"name": "toefl"})
    publication = IntegerField('Publications', render_kw={"name": "publication"})
    re_exp = IntegerField('Research Experience', render_kw={"name": "re_exp"})
    work_exp = IntegerField('Work Experience', render_kw={"name": "work_exp"})
    course = StringField('Course', render_kw={"name": "course"})  # Assuming course is a string
    scale = IntegerField('Scale', render_kw={"name": "scale"})
    degree = IntegerField('Degree', render_kw={"name": "degree"})
    university = StringField('University', render_kw={"name": "university"})
    submit = SubmitField('Predict')  # Submit button for form submission