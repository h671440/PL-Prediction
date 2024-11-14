from flask import Flask, render_template
import pandas as pd

#initialiser appen
app = Flask(__name__)

#definere hjemrute
@app.route('/')
def league_table():  
    #laste den simulerte liga-tabellen
    league_table_df = pd.read_csv("../data/league_table_prediction.csv")
    
   # Add a column for team logos without inline styles
    league_table_df['Team'] = league_table_df['Team'].apply(
        lambda x: f"<div class='team-logo'><img src='/static/logos/{x}.png' alt='{x}'> {x}</div>"
    )


   # Convert data to HTML with Bootstrap classes
    league_table_html = league_table_df.to_html(
        index=False, 
        escape=False,  # Allow HTML for the logo column
        classes="table table-striped table-bordered", 
        border=0
    )
    return render_template("index.html", table=league_table_html)


if __name__ == '__main__':
    app.run(debug=True)
