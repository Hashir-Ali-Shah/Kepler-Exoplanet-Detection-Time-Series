import os 
import tensorflow as tf 
CHECKPOINT_DIR ="/kaggle/working/checkpoints"
RESULTS_DIR ="/kaggle/working/results"
focal_loss =tf .keras .losses .BinaryFocalCrossentropy (gamma =2.0 ,from_logits =False )
INPUT_SHAPE =(3197 ,1 )
def build_lstm (input_shape ):
    from tensorflow .keras .models import Sequential 
    from tensorflow .keras .layers import LSTM ,Dense ,Dropout ,BatchNormalization 
    model =Sequential ([
    LSTM (128 ,return_sequences =True ,input_shape =input_shape ),
    BatchNormalization (),Dropout (0.4 ),
    LSTM (64 ,return_sequences =True ),
    BatchNormalization (),Dropout (0.4 ),
    LSTM (32 ,return_sequences =True ),
    BatchNormalization (),Dropout (0.4 ),
    LSTM (32 ,return_sequences =False ),
    BatchNormalization (),
    Dense (1 ,activation ="sigmoid"),
    ])
    return model 
def build_gru (input_shape ):
    from tensorflow .keras .models import Sequential 
    from tensorflow .keras .layers import GRU ,Dense ,Dropout ,BatchNormalization 
    model =Sequential ([
    GRU (128 ,return_sequences =True ,input_shape =input_shape ),
    BatchNormalization (),Dropout (0.4 ),
    GRU (64 ,return_sequences =True ),
    BatchNormalization (),Dropout (0.4 ),
    GRU (32 ,return_sequences =True ),
    BatchNormalization (),Dropout (0.4 ),
    GRU (32 ,return_sequences =False ),
    BatchNormalization (),
    Dense (1 ,activation ="sigmoid"),
    ])
    return model 
MODELS =[
{"name":"LSTM_SMOTE","builder":build_lstm ,"ckpt":"lstm_smote"},
{"name":"GRU_SMOTE","builder":build_gru ,"ckpt":"gru_smote"},
{"name":"LSTM_Windowed","builder":build_lstm ,"ckpt":"lstm_windowed"},
{"name":"GRU_Windowed","builder":build_gru ,"ckpt":"gru_windowed"},
]
os .makedirs (RESULTS_DIR ,exist_ok =True )
for cfg in MODELS :
    weights_path =os .path .join (CHECKPOINT_DIR ,cfg ["ckpt"],"weights.weights.h5")
    save_path =os .path .join (RESULTS_DIR ,f"{cfg['name']}.keras")
    if not os .path .exists (weights_path ):
        print (f"[{cfg['name']}] ⚠️  No weights found at {weights_path} — skipping.")
        continue 
    print (f"[{cfg['name']}] Loading weights...")
    model =cfg ["builder"](INPUT_SHAPE )
    model .compile (
    optimizer =tf .keras .optimizers .Adam (learning_rate =0.001 ),
    loss =focal_loss ,
    metrics =["accuracy",tf .keras .metrics .AUC (name ="auc")],
    )
    model .load_weights (weights_path )
    model .save (save_path )
    print (f"[{cfg['name']}] ✅  Saved → {save_path}")
print ("\nDone. All models saved in:",RESULTS_DIR )