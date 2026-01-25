import ast,sys
p = r'c:\Users\iProg\OneDrive\Documents\Football_predict\nfl_prediction_system\NFL_ML_Predictions\backend\pipeline_enhanced.py'
try:
    s = open(p,'r',encoding='utf-8').read()
    ast.parse(s)
    print('SYNTAX_OK')
except Exception:
    print('SYNTAX_ERR')
    import traceback; traceback.print_exc()
    sys.exit(1)
