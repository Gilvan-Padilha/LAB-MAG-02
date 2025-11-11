import numpy as np
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from unidecode import unidecode
import math
import scipy.special as sp
import reedsolo # Garanta que 'pip install reedsolo' foi executado

# --- Constantes Físicas ---
C = 299792458  # Velocidade da luz (m/s)
K_BOLTZMANN = 1.380649e-23  # Constante de Boltzmann (J/K)
T_PADRAO = 290           # Temperatura padrão (Kelvin, ~17°C)

# --- Constantes de Simulação ---
AMOSTRAS_POR_BIT = 50 
K_RS = 223 # Número de bytes de dados no Reed-Solomon
N_RS = 255 # Tamanho total do bloco (k + 32 bytes de paridade)
PARITY_BYTES = N_RS - K_RS # 32 bytes de paridade

# --- Inicializa o Codificador Reed-Solomon ---
RSC = reedsolo.RSCodec(PARITY_BYTES) 

# --- Inicialização do Servidor ---
app = Flask(__name__)
CORS(app) 

# ==========================================================
# FUNÇÕES DE UTILIDADE
# ==========================================================
def criar_sinal_digital(bits_array):
    sinal_digital = np.repeat(bits_array, AMOSTRAS_POR_BIT)
    return sinal_digital

# ==========================================================
# ROTA PARA SERVIR O SITE (index.html)
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')

# ==========================================================
# FUNÇÕES DE CÁLCULO DE ANTENA (Sem mudanças)
# ==========================================================
def calcular_parametros_antena(frequencia, comprimento_antena, tipo_antena, misalignment_angle):
    if frequencia == 0:
        return {"ganho_dbi": -20.0, "impedancia": "N/A", "is_resonant": False, "relacao_onda": "N/A", "diagrama_theta": [], "diagrama_r_db": []}
    lambda_c = C / frequencia
    comprimento_eletrico = comprimento_antena / lambda_c
    if tipo_antena == 'dipolo':
        fator_ressonancia = np.exp(-((comprimento_eletrico - 0.48)**2) / 0.05)
        max_ganho_dbi = -5.0 + (2.15 - (-5.0)) * fator_ressonancia
        is_resonant = fator_ressonancia > 0.7
        impedancia = f"~73 + j42 Ω (Ressonante)" if is_resonant else f"~{200 + 500*(1-fator_ressonancia):.0f} - j{300*(1-fator_ressonancia):.0f} Ω (Não Ressonante)"
    elif tipo_antena == 'yagi':
        fator_ressonancia = np.exp(-((comprimento_eletrico - 0.5)**2) / 0.1)
        max_ganho_dbi = -10.0 + (10.0 - (-10.0)) * fator_ressonancia 
        is_resonant = fator_ressonancia > 0.7
        impedancia = "~50 Ω (Ressonante)" if is_resonant else "Alta (Não Ressonante)"
    elif tipo_antena == 'circular': 
        max_ganho_dbi = 3.0
        impedancia = "~73 Ω"
        is_resonant = True
    elif tipo_antena == 'patch':
        fator_ressonancia = np.exp(-((comprimento_eletrico - 0.48)**2) / 0.05)
        max_ganho_dbi = 1.0 + (7.0 - 1.0) * fator_ressonancia 
        is_resonant = fator_ressonancia > 0.7
        impedancia = "~50 Ω (Ressonante)" if is_resonant else "Alta (Não Ressonante)"
    else: 
        max_ganho_dbi = 0.0
        impedancia = "N/A"
        is_resonant = False
    
    if tipo_antena == 'yagi' or tipo_antena == 'custom_linear':
        tipo_diagrama = 'yagi'
    elif tipo_antena == 'circular' or tipo_antena == 'custom_circular':
        tipo_diagrama = 'circular'
    else:
        tipo_diagrama = tipo_antena
        
    theta_graus, r_db = gerar_diagrama_radiacao(max_ganho_dbi, tipo_diagrama)

    try:
        if tipo_antena == 'circular' or tipo_antena == 'custom_circular':
            actual_ganho_dbi_no_angulo = max_ganho_dbi
        else:
            angle_float = float(misalignment_angle)
            theta_array = np.array(theta_graus)
            index = np.argmin(np.abs(theta_array - angle_float))
            actual_ganho_dbi_no_angulo = r_db[index]
    except Exception as e:
        print(f"Erro ao encontrar ganho no ângulo: {e}")
        actual_ganho_dbi_no_angulo = -99.0 
    return {
        "ganho_dbi": float(actual_ganho_dbi_no_angulo), 
        "impedancia": impedancia,
        "is_resonant": bool(is_resonant),
        "relacao_onda": f"{(comprimento_eletrico):.2f}λ",
        "diagrama_theta": theta_graus, 
        "diagrama_r_db": r_db         
    }

def gerar_diagrama_radiacao(max_ganho_dbi, tipo_antena):
    # (Sem mudanças)
    ganho_linear = 10**(max_ganho_dbi / 10)
    theta = np.linspace(0, 2 * np.pi, 360)
    if tipo_antena == 'dipolo':
        r = ganho_linear * (np.sin(theta)**2) 
    elif tipo_antena == 'yagi':
        r = ganho_linear * (np.cos(theta/2)**20 + 0.05 * np.sin(theta/2)**2)
    elif tipo_antena == 'circular':
        r = np.full(360, ganho_linear)
    elif tipo_antena == 'patch':
        r = ganho_linear * (np.cos(theta/2)**4)
    else:
        r = np.full(360, ganho_linear) 
    r_db = 10 * np.log10(r + 1e-3) 
    r_db = r_db - np.max(r_db) + max_ganho_dbi 
    r_db[r_db < -30] = -30
    theta_graus = np.rad2deg(theta)
    return theta_graus.tolist(), r_db.tolist()

# ==========================================================
# FUNÇÃO DE ENLACE (Sem mudanças)
# ==========================================================
def calcular_hata_path_loss(freq_mhz, dist_km, ht, hr, city_size):
    # (Sem mudanças)
    f = max(150, min(1500, freq_mhz))
    d = max(1, min(20, dist_km))
    ht_hata = max(30, min(200, ht))
    hr_hata = max(1, min(10, hr))
    if city_size == 'large':
        C_hr = (1.1 * math.log10(f) - 0.7) * hr_hata - (1.56 * math.log10(f) - 0.8)
    else: 
        C_hr = (0.8 + (1.1 * math.log10(f) - 0.7) * hr_hata - (1.56 * math.log10(f)))
    A = 69.55 + 26.16 * math.log10(f) - 13.82 * math.log10(ht_hata) - C_hr
    B = 44.9 - 6.55 * math.log10(ht_hata)
    L_hata_urban = A + B * math.log10(d)
    return L_hata_urban

def calcular_enlace(freq_hz, dist_m, ganho_tx_dbi, ganho_rx_dbi, potencia_tx_dbm, 
                    tipo_antena_tx, tipo_antena_rx, 
                    custom_pol_tx, custom_pol_rx,
                    polarization_angle_tx, polarization_angle_rx, 
                    perdas_adicionais_db, ht, hr,
                    propagation_model, city_size):
    # (Sem mudanças)
    if freq_hz == 0:
        return {"potencia_recebida_dbm": -999.0, "perda_propagacao_db": 0.0, "perdas_adicionais_db": 0.0, "perda_polarizacao_db": 0.0}
    
    lambda_c = C / freq_hz
    if dist_m < 0.01: dist_m = 0.01
    
    perda_pol_db = 0.0
    is_tx_linear = False
    if tipo_antena_tx in ['dipolo', 'yagi', 'patch']:
        is_tx_linear = True
    elif tipo_antena_tx == 'custom':
        is_tx_linear = (custom_pol_tx == 'linear')

    is_rx_linear = False
    if tipo_antena_rx in ['dipolo', 'yagi', 'patch']:
        is_rx_linear = True
    elif tipo_antena_rx == 'custom':
        is_rx_linear = (custom_pol_rx == 'linear')

    if is_tx_linear and is_rx_linear:
        delta_angle_rad = np.deg2rad(float(polarization_angle_tx) - float(polarization_angle_rx))
        cos_delta = np.cos(delta_angle_rad)
        if np.abs(cos_delta) < 1e-9:
            perda_pol_db = 200.0
        else:
            perda_pol_db = -20 * np.log10(np.abs(cos_delta))
    elif is_tx_linear != is_rx_linear: 
        perda_pol_db = 3.0

    perda_propagacao_db = 0.0
    if propagation_model == 'friis':
        perda_propagacao_db = -20 * np.log10(lambda_c / (4 * np.pi * dist_m))
    elif propagation_model == 'tworay':
        try:
            d_los = math.sqrt(dist_m**2 + (ht - hr)**2)
            d_gr = math.sqrt(dist_m**2 + (ht + hr)**2)
            delta_d = d_gr - d_los
            delta_phi = (2 * np.pi * delta_d) / lambda_c 
            A = d_los / d_gr
            fator_linear = (1 - A * np.cos(delta_phi))**2 + (-A * np.sin(delta_phi))**2
            if fator_linear < 1e-20:
                 gain_solo_db = -200.0
            else:
                 gain_solo_db = 10 * np.log10(fator_linear)
            perda_los_db = -20 * np.log10(lambda_c / (4 * np.pi * d_los))
            perda_propagacao_db = perda_los_db - gain_solo_db
        except Exception as e:
            print(f"Erro no cálculo de 2 raios: {e}")
            perda_propagacao_db = -20 * np.log10(lambda_c / (4 * np.pi * dist_m))
    elif propagation_model == 'hata_urban':
        freq_mhz = freq_hz / 1e6
        dist_km = dist_m / 1e3
        perda_propagacao_db = calcular_hata_path_loss(freq_mhz, dist_km, ht, hr, city_size)
    else: 
        perda_propagacao_db = -20 * np.log10(lambda_c / (4 * np.pi * dist_m))
        
    potencia_recebida_dbm = (
        float(potencia_tx_dbm) + 
        float(ganho_tx_dbi) + 
        float(ganho_rx_dbi) - 
        perda_propagacao_db -
        perda_pol_db -
        float(perdas_adicionais_db)
    )
    
    return {
        "potencia_recebida_dbm": float(potencia_recebida_dbm),
        "perda_propagacao_db": float(perda_propagacao_db),
        "perdas_adicionais_db": float(perdas_adicionais_db),
        "perda_polarizacao_db": float(perda_pol_db)
    }

# ==========================================================
# FUNÇÕES DE MODULAÇÃO (Sem mudanças)
# ==========================================================

def calcular_ber(modulacao, eb_n0_linear):
    # (Sem mudanças)
    try:
        if eb_n0_linear <= 0: return 0.5
        if modulacao == 'bpsk': return 0.5 * sp.erfc(math.sqrt(eb_n0_linear))
        elif modulacao == 'qpsk': return 0.5 * sp.erfc(math.sqrt(eb_n0_linear))
        elif modulacao == 'qam16':
            es_n0_linear = 4 * eb_n0_linear
            ser = 3.0 * 0.5 * sp.erfc(math.sqrt(es_n0_linear / 10.0))
            ber = ser / 4.0
            return ber
        else: return 0.5 * sp.erfc(math.sqrt(eb_n0_linear / 2.0))
    except Exception as e:
        print(f"Erro no cálculo do BER: {e}")
        return 0.5

def gerar_constelacao(modulacao, snr_linear, num_pontos=200):
    # (Sem mudanças)
    if modulacao == 'bpsk':
        pontos_ideais = np.array([-1, 1]); indices = np.random.randint(0, 2, num_pontos); ideal_i = pontos_ideais[indices]; ideal_q = np.zeros(num_pontos)
    elif modulacao == 'qpsk':
        pontos_ideais = 1/np.sqrt(2) * np.array([1+1j, 1-1j, -1+1j, -1-1j]); indices = np.random.randint(0, 4, num_pontos); ideal_i = pontos_ideais[indices].real; ideal_q = pontos_ideais[indices].imag
    elif modulacao == 'qam16':
        fator = 1 / np.sqrt(10); pontos_i = np.array([-3, -1, 1, 3]) * fator; pontos_q = np.array([-3, -1, 1, 3]) * fator; indices_i = np.random.randint(0, 4, num_pontos); indices_q = np.random.randint(0, 4, num_pontos); ideal_i = pontos_i[indices_i]; ideal_q = pontos_q[indices_q]
    else: 
        return {"i": [], "q": []}
    if snr_linear < 1e-10: sigma = 1e6 
    else: sigma = math.sqrt(1 / (2 * snr_linear))
    noise_i = np.random.normal(0, sigma, num_pontos); noise_q = np.random.normal(0, sigma, num_pontos)
    rx_i = ideal_i + noise_i; rx_q = ideal_q + noise_q
    return {"i": rx_i.tolist(), "q": rx_q.tolist()}

# ==========================================================
# ROTA PRINCIPAL DA API (ATUALIZADA)
# ==========================================================
@app.route('/calcular_simulacao', methods=['POST'])
def calcular_simulacao():
    # --- Padrões ---
    snr_db = 0.0
    ber = 0.5
    potencia_recebida_media_dbm = -999.0
    potencia_recebida_fading_dbm = -999.0
    perda_propagacao_db = 0.0
    perdas_adicionais_db = 0.0
    perda_polarizacao_db = 0.0
    mensagem_recebida = "???"
    mensagem = "A"
    lambda_c = 0.0
    params_antena_tx = {"ganho_dbi": 0, "impedancia": "Erro", "is_resonant": False, "relacao_onda": "Erro"}
    params_antena_rx = {"ganho_dbi": 0, "impedancia": "Erro", "is_resonant": False, "relacao_onda": "Erro"}
    diagrama_data_tx = {"theta": [], "r_db": []}
    diagrama_data_rx = {"theta": [], "r_db": []}
    plot_data_distancia = {"distancias_km": [], "potencias_dbm": []}
    bits_para_transmitir = np.array([0])
    t_signal = np.array([0])
    signal_tx_modulado_ideal = np.array([0])
    signal_rx_modulado_com_ruido = np.array([0])
    bits_desmodulados_plot = np.array([0])
    amplitude_rx_v = 0.0
    amplitude_noise_v = 0.0
    plot_data_constellation = {"i": [], "q": []}
    
    try:
        # 1. Obter dados do frontend
        data = request.json
        freq_mhz = float(data.get('frequencia', 0))
        distancia_km = float(data.get('distancia', 0))
        mensagem = data.get('message', 'A')
        
        largura_banda_mhz = float(data.get('largura_banda_mhz', 1.0))
        noise_figure_db = float(data.get('noise_figure_db', 5.0))
        
        use_fec = data.get('use_fec', False)
        modulacao = data.get('modulacao', 'bpsk')
        perdas_adicionais_db = float(data.get('perdas_adicionais_db', 0.0))
        potencia_tx_dbm = float(data.get('potencia_tx_dbm', 20.0))
        snr_min_db = float(data.get('snr_min_db', 12.0))
        
        comp_antena_m_tx = float(data.get('comprimento_antena_tx', 0))
        tipo_antena_tx = data.get('antenna_type_tx', 'dipolo')
        misalignment_angle_tx = float(data.get('misalignment_angle_tx', 0.0))
        polarization_angle_tx = float(data.get('polarization_angle_tx', 0.0))
        ganho_manual_tx_dbi = float(data.get('ganho_manual_tx_dbi', 0.0))
        custom_pol_tx = data.get('custom_pol_tx', 'linear')

        comp_antena_m_rx = float(data.get('comprimento_antena_rx', 0))
        tipo_antena_rx = data.get('antenna_type_rx', 'dipolo')
        misalignment_angle_rx = float(data.get('misalignment_angle_rx', 0.0))
        polarization_angle_rx = float(data.get('polarization_angle_rx', 0.0))
        ganho_manual_rx_dbi = float(data.get('ganho_manual_rx_dbi', 0.0))
        custom_pol_rx = data.get('custom_pol_rx', 'linear')

        ht = float(data.get('ht', 1.5))
        hr = float(data.get('hr', 1.5))
        propagation_model = data.get('propagation_model', 'friis')
        city_size = data.get('city_size', 'small_medium')
        
        # --- NOVO: Dados de Fading ---
        fading_model = data.get('fading_model', 'none')
        rician_k_db = float(data.get('rician_k_db', 10.0))
        
        if not mensagem:
            mensagem = 'A'
        freq_hz = freq_mhz * 1e6
        dist_m = distancia_km * 1e3
        
        # 2. Fase 2: Simular as Antenas
        lambda_c = (C / freq_hz) if freq_hz > 0 else 0
        
        if tipo_antena_tx == 'custom':
            if custom_pol_tx == 'circular':
                tipo_diagrama_tx = 'custom_circular'
                misalignment_angle_tx = 0
            else:
                tipo_diagrama_tx = 'custom_linear'
            theta_graus_tx, r_db_tx = gerar_diagrama_radiacao(ganho_manual_tx_dbi, tipo_diagrama_tx)
            if custom_pol_tx == 'circular':
                actual_ganho_dbi_no_angulo_tx = ganho_manual_tx_dbi
            else:
                angle_float_tx = float(misalignment_angle_tx)
                theta_array_tx = np.array(theta_graus_tx)
                index_tx = np.argmin(np.abs(theta_array_tx - angle_float_tx))
                actual_ganho_dbi_no_angulo_tx = r_db_tx[index_tx]
            params_antena_tx = {
                "ganho_dbi": float(actual_ganho_dbi_no_angulo_tx), 
                "impedancia": "N/A (Manual)", "is_resonant": True, "relacao_onda": "N/A (Manual)",
                "diagrama_theta": theta_graus_tx, "diagrama_r_db": r_db_tx         
            }
            diagrama_data_tx = {"theta": theta_graus_tx, "r_db": r_db_tx}
        else:
            params_antena_tx = calcular_parametros_antena(
                freq_hz, comp_antena_m_tx, tipo_antena_tx, misalignment_angle_tx
            )
            diagrama_data_tx = {
                "theta": params_antena_tx['diagrama_theta'],
                "r_db": params_antena_tx['diagrama_r_db']
            }
        
        if tipo_antena_rx == 'custom':
            if custom_pol_rx == 'circular':
                tipo_diagrama_rx = 'custom_circular'
                misalignment_angle_rx = 0
            else:
                tipo_diagrama_rx = 'custom_linear'
            theta_graus_rx, r_db_rx = gerar_diagrama_radiacao(ganho_manual_rx_dbi, tipo_diagrama_rx)
            if custom_pol_rx == 'circular':
                actual_ganho_dbi_no_angulo_rx = ganho_manual_rx_dbi
            else:
                angle_float_rx = float(misalignment_angle_rx)
                theta_array_rx = np.array(theta_graus_rx)
                index_rx = np.argmin(np.abs(theta_array_rx - angle_float_rx))
                actual_ganho_dbi_no_angulo_rx = r_db_rx[index_rx]
            params_antena_rx = {
                "ganho_dbi": float(actual_ganho_dbi_no_angulo_rx), 
                "impedancia": "N/A (Manual)", "is_resonant": True, "relacao_onda": "N/A (Manual)",
                "diagrama_theta": theta_graus_rx, "diagrama_r_db": r_db_rx         
            }
            diagrama_data_rx = {"theta": theta_graus_rx, "r_db": r_db_rx}
        else:
            params_antena_rx = calcular_parametros_antena(
                freq_hz, comp_antena_m_rx, tipo_antena_rx, misalignment_angle_rx
            )
            diagrama_data_rx = {
                "theta": params_antena_rx['diagrama_theta'],
                "r_db": params_antena_rx['diagrama_r_db']
            }

        # 3. Fase 3a: Simular Propagação (Potência Média)
        dados_enlace = calcular_enlace(
            freq_hz=freq_hz, dist_m=dist_m, ganho_tx_dbi=params_antena_tx['ganho_dbi'],
            ganho_rx_dbi=params_antena_rx['ganho_dbi'], potencia_tx_dbm=potencia_tx_dbm,
            tipo_antena_tx=tipo_antena_tx, tipo_antena_rx=tipo_antena_rx,
            custom_pol_tx=custom_pol_tx, custom_pol_rx=custom_pol_rx,
            polarization_angle_tx=polarization_angle_tx, polarization_angle_rx=polarization_angle_rx,
            perdas_adicionais_db=perdas_adicionais_db, ht=ht, hr=hr,
            propagation_model=propagation_model, city_size=city_size
        )
        potencia_recebida_media_dbm = dados_enlace['potencia_recebida_dbm']
        perda_propagacao_db = dados_enlace['perda_propagacao_db']
        perdas_adicionais_db = dados_enlace['perdas_adicionais_db']
        perda_polarizacao_db = dados_enlace['perda_polarizacao_db']
        
        if math.isnan(potencia_recebida_media_dbm) or math.isinf(potencia_recebida_media_dbm):
            potencia_recebida_media_dbm = -999.0
            
        
        # --- MUDANÇA AQUI: Fase 3b - Aplicar Fading de Pequena Escala ---
        potencia_recebida_fading_dbm = potencia_recebida_media_dbm
        
        try:
            if fading_model != 'none':
                pr_media_watts = (10**(potencia_recebida_media_dbm / 10)) / 1000
                
                if fading_model == 'rayleigh':
                    # A potência em Rayleigh segue uma distribuição exponencial
                    # A média (scale) da distribuição é a potência média
                    fading_watts = np.random.exponential(scale=pr_media_watts)
                    if fading_watts < 1e-30: # Evita log(0)
                        potencia_recebida_fading_dbm = -999.0
                    else:
                        potencia_recebida_fading_dbm = 10 * np.log10(fading_watts * 1000)
                
                elif fading_model == 'rician':
                    k_linear = 10**(rician_k_db / 10)
                    
                    # Potência (Watts) do caminho dominante (LOS)
                    pr_dominante_watts = pr_media_watts * k_linear / (k_linear + 1)
                    # Potência (Watts) dos caminhos refletidos (NLOS)
                    pr_refletida_watts = pr_media_watts / (k_linear + 1)
                    
                    # Amplitude (RMS Volts, assumindo R=1) dos caminhos
                    v_rms_dominante = np.sqrt(pr_dominante_watts)
                    v_rms_refletida = np.sqrt(pr_refletida_watts)
                    
                    # Sigma da gaussiana (I e Q) para os caminhos refletidos
                    # pr_refletida_watts = 2 * sigma^2
                    sigma = v_rms_refletida / np.sqrt(2)
                    
                    # Gera os componentes I e Q
                    I = v_rms_dominante + np.random.normal(0, sigma)
                    Q = np.random.normal(0, sigma)
                    
                    # Amplitude RMS resultante (com fading)
                    v_rms_fading = np.sqrt(I**2 + Q**2)
                    
                    # Potência em Watts resultante
                    potencia_com_fading_watts = v_rms_fading**2
                    
                    if potencia_com_fading_watts < 1e-30:
                        potencia_recebida_fading_dbm = -999.0
                    else:
                        potencia_recebida_fading_dbm = 10 * np.log10(potencia_com_fading_watts * 1000)
        
        except Exception as e_fade:
            print(f"Erro no cálculo de fading: {e_fade}")
            potencia_recebida_fading_dbm = potencia_recebida_media_dbm
        # --- FIM DA MUDANÇA ---


        # 4. Fase 4: Calcular Desempenho (SNR, BER) e Ruído
        bandwidth_hz = largura_banda_mhz * 1e6
        noise_power_dbm = 10 * np.log10(K_BOLTZMANN * T_PADRAO * bandwidth_hz * 1000)
        total_noise_floor_dbm = noise_power_dbm + noise_figure_db
        
        # --- MUDANÇA AQUI: SNR usa a potência com FADING ---
        snr_db = potencia_recebida_fading_dbm - total_noise_floor_dbm
        # --- FIM DA MUDANÇA ---
        
        if math.isnan(snr_db) or math.isinf(snr_db): snr_db = 0.0
        snr_linear = 10**(snr_db / 10)
        ber = calcular_ber(modulacao, snr_linear)
        
        noise_watts_total = (10**(total_noise_floor_dbm / 10)) / 1000
        amplitude_noise_v = np.sqrt(noise_watts_total * 50) 
        if np.isnan(amplitude_noise_v) or np.isinf(amplitude_noise_v):
            amplitude_noise_v = 0.0

        # 5. Fase 1 e 4: Codificação, Simulação de Erros, Decodificação
        bytes_originais = mensagem.encode('utf-8', 'replace')[:K_RS]
        if use_fec:
            bytes_para_transmitir = RSC.encode(bytes_originais)
        else:
            bytes_para_transmitir = bytes_originais
        bits_para_transmitir = np.unpackbits(np.frombuffer(bytes_para_transmitir, dtype=np.uint8))
        erros = np.random.rand(len(bits_para_transmitir))
        indices_de_erro = (erros < ber)
        bits_corrompidos = np.copy(bits_para_transmitir)
        bits_corrompidos[indices_de_erro] = 1 - bits_corrompidos[indices_de_erro]
        resto = len(bits_corrompidos) % 8
        if resto != 0:
            bits_corrompidos = np.concatenate((bits_corrompidos, np.zeros(8 - resto, dtype=int)))
        bytes_corrompidos = np.packbits(bits_corrompidos).tobytes()
        if use_fec:
            try:
                bytes_reconstruidos, _, _ = RSC.decode(bytes_corrompidos)
                mensagem_recebida = bytes_reconstruidos.decode('utf-8', 'replace')
            except reedsolo.ReedSolomonError:
                mensagem_recebida = "!!! FALHA NA CORREÇÃO (MUITOS ERROS) !!!"
        else:
            mensagem_recebida = bytes_corrompidos[:len(bytes_originais)].decode('utf-8', 'replace')
            
        # 6a. Gerar Sinais para Gráficos
        plot_data_constellation = gerar_constelacao(modulacao, snr_linear)
        bits_para_plotar_ideal = bits_para_transmitir[:10]
        bits_desmodulados_plot = bits_corrompidos[:10]
        sinal_digital_tx_esticado = criar_sinal_digital(bits_para_plotar_ideal)
        if len(sinal_digital_tx_esticado) == 0:
            t_signal = np.array([0]); signal_tx_modulado_ideal = np.array([0]); signal_rx_modulado_com_ruido = np.array([0]); amplitude_rx_v = 0.0
        else:
            BIT_RATE_PLOT = 1000 
            duracao_total_s = len(bits_para_plotar_ideal) / BIT_RATE_PLOT
            num_amostras_total = len(sinal_digital_tx_esticado)
            t_signal = np.linspace(0, duracao_total_s, num_amostras_total)
            onda_portadora_v = np.sin(2 * np.pi * freq_hz * t_signal)
            amplitude_tx_v = 1.0
            signal_tx_modulado_ideal = onda_portadora_v * sinal_digital_tx_esticado * amplitude_tx_v
            
            # --- MUDANÇA AQUI: Usa a potência com FADING ---
            Pr_watts_fading = (10**(potencia_recebida_fading_dbm / 10)) / 1000
            amplitude_rx_v = np.sqrt(Pr_watts_fading * 50) 
            # --- FIM DA MUDANÇA ---

            if np.isnan(amplitude_rx_v) or amplitude_rx_v < 1e-9:
                amplitude_rx_v = 0.0
            sinal_digital_rx_esticado = criar_sinal_digital(bits_desmodulados_plot)
            sinal_rx_puro = onda_portadora_v * sinal_digital_rx_esticado * amplitude_rx_v
            ruido_no_sinal = np.random.normal(0, amplitude_noise_v, num_amostras_total)
            signal_rx_modulado_com_ruido = sinal_rx_puro + ruido_no_sinal
            
        # 6b. Gerar Plot de Distância (Usa a potência MÉDIA, o que está correto)
        NUM_PONTOS_PLOT_DIST = 500
        dist_steps_m = np.linspace(max(1.0, dist_m / NUM_PONTOS_PLOT_DIST), dist_m, NUM_PONTOS_PLOT_DIST)
        potencias_plot_dbm = []
        distancias_plot_km = (dist_steps_m / 1000.0).tolist()
        
        for d_step_m in dist_steps_m:
            dados_enlace_step = calcular_enlace(
                freq_hz=freq_hz, dist_m=d_step_m, ganho_tx_dbi=params_antena_tx['ganho_dbi'],
                ganho_rx_dbi=params_antena_rx['ganho_dbi'], potencia_tx_dbm=potencia_tx_dbm,
                tipo_antena_tx=tipo_antena_tx, tipo_antena_rx=tipo_antena_rx,
                custom_pol_tx=custom_pol_tx, custom_pol_rx=custom_pol_rx,
                polarization_angle_tx=polarization_angle_tx, polarization_angle_rx=polarization_angle_rx,
                perdas_adicionais_db=perdas_adicionais_db, ht=ht, hr=hr,
                propagation_model=propagation_model, city_size=city_size
            )
            potencias_plot_dbm.append(dados_enlace_step['potencia_recebida_dbm'])
            
        plot_data_distancia = {"distancias_km": distancias_plot_km, "potencias_dbm": potencias_plot_dbm}

    except Exception as e:
        print(f"[Erro app.py] Erro principal na simulação: {e}")
        snr_db, ber = 0.0, 0.5
        potencia_recebida_media_dbm = -999.0
        potencia_recebida_fading_dbm = -999.0
        perda_propagacao_db, perdas_adicionais_db, perda_polarizacao_db = 0.0, 0.0, 0.0
        params_antena_tx = {"ganho_dbi": 0, "impedancia": "Erro", "is_resonant": False, "relacao_onda": "Erro"}
        params_antena_rx = {"ganho_dbi": 0, "impedancia": "Erro", "is_resonant": False, "relacao_onda": "Erro"}
        lambda_c, diagrama_data_tx, diagrama_data_rx = 0.0, {"theta": [], "r_db": []}, {"theta": [], "r_db": []}
        mensagem_recebida = "ERRO NO SERVIDOR"
        bits_para_transmitir = np.array([0])
        plot_data_distancia = {"distancias_km": [], "potencias_dbm": []}
        
        largura_banda_mhz = 1.0
        noise_figure_db = 5.0
        noise_power_dbm = 10 * np.log10(K_BOLTZMANN * T_PADRAO * (largura_banda_mhz * 1e6) * 1000)
        total_noise_floor_dbm = noise_power_dbm + noise_figure_db
        snr_min_db = 12.0
        potencia_tx_dbm = 20.0
        tipo_antena_tx = 'dipolo'
        tipo_antena_rx = 'dipolo'
        custom_pol_tx = 'linear'
        custom_pol_rx = 'linear'
        fading_model = 'none'
        rician_k_db = 10.0


    # 7. Preparar dados de retorno (MUDANÇA AQUI)
    return_data = {
        "wavelength_m": lambda_c,
        "antenna_params_tx": params_antena_tx,
        "antenna_params_rx": params_antena_rx,
        "propagation_params": {
            "potencia_recebida_media_dbm": potencia_recebida_media_dbm, # Para o Link Budget
            "potencia_recebida_dbm": potencia_recebida_fading_dbm,     # Para o SNR e Resultados
            "potencia_recebida_v": amplitude_rx_v,
            "amplitude_noise_v": amplitude_noise_v,
            "snr_db": snr_db,
            "ber": ber,
            "lprop_db": perda_propagacao_db,
            "latm_db": perdas_adicionais_db,
            "lpol_db": perda_polarizacao_db,
        },
        "sim_params": {
            "total_noise_floor_dbm": total_noise_floor_dbm,
            "largura_banda_mhz": largura_banda_mhz,
            "noise_figure_db": noise_figure_db,
            "snr_min_db": snr_min_db,
            "potencia_tx_dbm": potencia_tx_dbm,
            "antenna_type_tx": tipo_antena_tx,
            "antenna_type_rx": tipo_antena_rx,
            "custom_pol_tx": custom_pol_tx,
            "custom_pol_rx": custom_pol_rx,
            "fading_model": fading_model,
            "rician_k_db": rician_k_db
        },
        "plot_data": {
            "radiation_pattern": diagrama_data_tx, 
            "radiation_pattern_rx": diagrama_data_rx,
            "signal_tx_modulado_ideal": {"t_s": t_signal.tolist(), "signal_v": signal_tx_modulado_ideal.tolist()},
            "signal_rx_modulado_com_ruido": {"t_s": t_signal.tolist(), "signal_v": signal_rx_modulado_com_ruido.tolist()},
            "bits_desmodulados_plot": {
                "t_s_bits": np.arange(len(bits_desmodulados_plot)).tolist(),
                "bits": bits_desmodulados_plot.tolist()
            },
            "constellation": plot_data_constellation,
            "distancia_plot": plot_data_distancia
        },
        "original_message": mensagem,
        "received_message": mensagem_recebida
    }
    
    print(f"Mod={modulacao}, Fading={fading_model}, Pr_media={potencia_recebida_media_dbm:.2f} dBm, Pr_inst={potencia_recebida_fading_dbm:.2f} dBm, SNR={snr_db:.2f} dB, BER={ber:.2e}")

    return jsonify(return_data)

# --- Executar o Servidor ---
if __name__ == '__main__':
    print("Iniciando servidor Flask.")
    print("Acesse o simulador em: http://127.0.0.1:5000/")
    app.run(debug=True, port=5000)