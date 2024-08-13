# AI Challenge: Evaluate Student Summaries
This group project is part of the "AI Challenge," a 12 ECTS course offered by the Lucerne University of Applied Sciences. In this course a [final report](EvalStudentSummaries_Report.pdf) will be created and evaluated

Team Members:
- Jonathan Carona
- Tharrmeehan Krishnathasan
- Josef Rittiner

Throughout this course, our team participated in the already finished Kaggle competition titled <a href="https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries"> "Evaluate Student Summaries." </a> The primary objective of this competition is as follows:

""
The goal of this competition is to assess the quality of summaries written by students in grades 3-12. You'll build a model that evaluates how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary. You'll have access to a collection of real student summaries to train your model. 

Your work will assist teachers in evaluating the quality of student work and also help learning platforms provide immediate feedback to students.
""
(Alex Franklin, asiegel, HCL-Jevster, Jules King, julianmante, Maggie, Perpetual Baffour, Ryan Holbrook, Scott Crossley. (2023). CommonLit - Evaluate Student Summaries. Kaggle. https://kaggle.com/competitions/commonlit-evaluate-student-summaries)

The team developed three distinct models:
The ROUGE-Based Model is a Neural Network, which uses ROUGE metrics as it’s input
features. The LightGBM Model uses an array of scores, which can determine readability,
complexity and the grade level of a summary. The DeBerta Transformer, a state-of-the-art
language model which balances out the two while also considering the specific task the stu-
dents were asked to capture in their summary.
These three models collectively form the final Ensemble Model. Each model independently
preprocesses the texts and generates predictions. The Ensemble Model weighs the contri-
butions of each model to produce a final score. The results showcase respectable perfor-
mance in the Kaggle competition, placing 975th out of a total 2’065 participants.

For further information please read our [report](EvalStudentSummaries_Report.pdf)


