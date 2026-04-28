to init:

model = TDNNPredictor(device= "cpu" ,weight_path = "dl_model/speechbrain_ablation/checkpoints/tdnn_full_best_acc.pth") 
or 
model = TDNNPredictor() #Default as ↑

to predict:

is_switch, confidence = model.predict(audio_first, audio_second) # audio1: np.ndarray, audio2: np.ndarray
is_switch=True->codeswitch False->change speaker