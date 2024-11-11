from flask import Flask, render_template
import pandas as pd

#initialiser appen
app = Flask(__name__)

#definere hjemrute
@app.route('/')
def league_table():  
    #laste den simulerte liga-tabellen
    league_table_df = pd.read_csv("../data/league_table_simulation.csv")
    
 # Convert data to HTML with Bootstrap classes
    league_table_html = league_table_df.to_html(index=False, classes="table table-striped table-bordered", border=0)

    return render_template("index.html", table=league_table_html)


if __name__ == '__main__':
    app.run(debug=True)
