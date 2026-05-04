from flask import Flask, request, jsonify, send_file
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

def rain_coeff(f):
    if f < 10:
        return 0.0001*f**2, 1
    elif f < 20:
        return 0.0003*f**2, 1.1
    else:
        return 0.0008*f**2, 1.3

@app.route('/')
def home():
    return send_file('final_ui.html')

@app.route('/run', methods=['POST'])
def run_model():
    
    # 🔥 1) RANDOM FIX
    np.random.seed(0)

    data = request.json
    
    # 🔥 4) FLOAT ANIQLIK
    f = np.float64(data['f'])
    theta = np.float64(data['theta'])
    R_input = np.float64(data['R'])
    M_input = np.float64(data['M'])
    target = np.float64(data['target'])/100
    
    Pt, Gt, Gr = 20, 30, 30
    SNR_min = 10
    Ntime = 365
    
    c = 3e8
    d = 3.6e7
    
    snr_time = []
    
    # ===== ASOSIY MODEL =====
    for _ in range(Ntime):
        
        # 🔥 2) MATLAB RANDN GA YAQIN
        R = max(0, R_input + 0.3*R_input*np.random.standard_normal())
        
        L_fs = 20*np.log10(4*np.pi*d*(f*1e9)/c)
        k, alpha = rain_coeff(f)
        L_rain = k * R**alpha
        L_cloud = (M_input / np.sin(np.radians(theta))) * (0.08*f)
        L_gas = 0.02*f + 0.0003*f**2
        
        L_total = L_fs + L_rain + L_cloud + L_gas
        
        Pr = Pt + Gt + Gr - L_total
        N = -228.6 + 10*np.log10(290) + 10*np.log10(1e6)
        
        snr = Pr - N + np.random.standard_normal()*2
        snr_time.append(snr)
    
    snr_time = np.array(snr_time)
    
    outage = np.mean(snr_time < SNR_min)
    availability = 1 - outage
    
    # ===== OPTIMAL SHAROIT =====
    best_diff = 1
    best_R = best_f = best_M = 0
    
    # 🔥 3) MATLAB GRID
    M_grid = np.linspace(0,2,11)
    
    for f_test in range(5,41):
        for R_test in range(0,31):
            for M_test in M_grid:
                
                success = 0
                
                for _ in range(100):
                    
                    R_rand = max(0, R_test + 0.3*R_test*np.random.standard_normal())
                    
                    L_fs = 20*np.log10(4*np.pi*d*(f_test*1e9)/c)
                    k, alpha = rain_coeff(f_test)
                    L_rain = k * R_rand**alpha
                    L_cloud = (M_test / np.sin(np.radians(theta))) * (0.08*f_test)
                    L_gas = 0.02*f_test + 0.0003*f_test**2
                    
                    L_total = L_fs + L_rain + L_cloud + L_gas
                    
                    Pr = Pt + Gt + Gr - L_total
                    snr = Pr - N
                    
                    if snr > SNR_min:
                        success += 1
                
                availability_test = success/100
                diff = abs(availability_test - target)
                
                if diff < best_diff:
                    best_diff = diff
                    best_R = R_test
                    best_f = f_test
                    best_M = M_test
    
    # ===== BERILGAN SHAROIT UCHUN CHASTOTA =====
    best_diff_f = 1
    best_freq = 0
    best_av = 0
    
    for f_test in np.arange(5,40.5,0.5):
        
        success = 0
        
        for _ in range(200):
            
            R_rand = max(0, R_input + 0.3*R_input*np.random.standard_normal())
            
            L_fs = 20*np.log10(4*np.pi*d*(f_test*1e9)/c)
            k, alpha = rain_coeff(f_test)
            L_rain = k * R_rand**alpha
            L_cloud = (M_input / np.sin(np.radians(theta))) * (0.08*f_test)
            L_gas = 0.02*f_test + 0.0003*f_test**2
            
            L_total = L_fs + L_rain + L_cloud + L_gas
            
            Pr = Pt + Gt + Gr - L_total
            snr = Pr - N
            
            if snr > SNR_min:
                success += 1
        
        av = success/200
        diff = abs(av - target)
        
        if diff < best_diff_f:
            best_diff_f = diff
            best_freq = f_test
            best_av = av
    
    # ===== GRAFIK =====
    if not os.path.exists("static"):
        os.mkdir("static")
    
    plt.figure()
    plt.plot(snr_time)
    plt.title("SNR vaqt bo‘yicha")
    plt.savefig("static/snr.png")
    plt.close()
    
    plt.figure()
    plt.hist(snr_time, bins=20)
    plt.title("SNR taqsimoti")
    plt.savefig("static/hist.png")
    plt.close()
    
    return jsonify({
        "availability": round(availability*100,2),
        "best_f": best_f,
        "best_R": best_R,
        "best_M": round(best_M,2),
        "freq_opt": round(best_freq,2),
        "freq_av": round(best_av*100,2)
    })

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)