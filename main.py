from db.database import db
import utils.collectData as data
import model.trainModel as train

# data.getAllYesterdayPlayerMatches()
# data.getAllMatchDetails()
train.train_on_new_matches()
train.validate_on_new_matches()

if db.client:
    db.client.close()
