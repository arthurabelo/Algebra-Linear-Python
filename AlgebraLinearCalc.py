import tkinter as tk
import numpy as np
import re
from tkinter import messagebox
from tkinter import ttk
from fractions import Fraction
import math
import sympy as sp


class Base:
    def __init__(self, vetores):
        self.vetores = np.array(vetores, dtype=float)
        self.dim = self.vetores.shape[1] if len(self.vetores.shape) > 1 else 1

    def matriz_para_base(self, nova_base):
        # Retorna a matriz de mudança de base de self para nova_base
        if np.linalg.matrix_rank(self.vetores) < self.vetores.shape[1] or np.linalg.matrix_rank(nova_base.vetores) < nova_base.vetores.shape[1]:
            raise ValueError("Uma das bases fornecidas não é invertível (vetores linearmente dependentes).")
        return np.linalg.inv(self.vetores) @ nova_base.vetores


class TransformacaoLinear:
    def __init__(self, incognitas, expressoes):
        self.incognitas = incognitas
        self.expressoes = expressoes
        self.matriz = self.matriz_associada(incognitas, expressoes)

    def matriz_associada(self, incognitas, expressoes):
        matriz = []
        for expr in expressoes:
            matriz.append(self.expr_para_coef(expr, incognitas))
        return np.array(matriz)
    
    @staticmethod
    def expr_para_coef(expr, incognitas):
        coefs = [0.0 for _ in incognitas]
        expr = expr.replace(' ', '')
        termos = re.findall(r'[+-]?[^+-]+', expr)
        for termo in termos:
            if termo == '':
                continue
            achou = False
            for i, var in enumerate(incognitas):
                if var in termo:
                    achou = True
                    coef = termo.replace(var, '')
                    if coef in ['', '+']:
                        coefs[i] += 1.0
                    elif coef == '-':
                        coefs[i] -= 1.0
                    else:
                        coefs[i] += float(coef)
        return coefs

    @staticmethod
    def parse_transformacao(transformacao):
        def extrair_lista(regex, texto):
            match = re.search(regex, texto)
            if not match:
                return []
            lista_str = match.group(1)
            lista = []
            for item in lista_str.split(','):
                item = item.strip()
                if item == '':
                    continue
                lista.append(item)
            return lista
        try:
            incognitas = extrair_lista(r'T\s*\(\s*([^)]*?)\s*\)\s*', transformacao)
            expressoes = extrair_lista(r'=\s*\(\s*([^)]*?)\s*\)\s*', transformacao)
            if not incognitas or not expressoes:
                raise ValueError
            return incognitas, expressoes
        except Exception:
            return [], []

    def matriz_em_bases(self, base_dom, base_cod, opcao_menu=None):
        P = base_dom.vetores
        Q = base_cod.vetores
        if opcao_menu == '3':
            if self.matriz.shape[0] != self.matriz.shape[1]:
                raise ValueError("A matriz associada não é quadrada. Autovalores e autovetores só existem para matrizes quadradas.")
        if opcao_menu == '2':
            if P.shape[0] != len(self.incognitas):
                raise ValueError(f"A base do domínio deve ter dimensão {len(self.incognitas)} (número de variáveis de entrada).")
            if Q.shape[0] != len(self.expressoes):
                raise ValueError(f"A base do contradomínio deve ter dimensão {len(self.expressoes)} (número de variáveis de saída).")
        
        if Q.shape[0] == Q.shape[1]:
            Q_inv = np.linalg.inv(Q)
        else:
            Q_inv = np.linalg.pinv(Q)
        if P.shape[0] == P.shape[1]:
            P_inv = np.linalg.inv(P)
        else:
            P_inv = np.linalg.pinv(P)
        return Q_inv @ self.matriz @ P

    def eliminacao_gauss(self, M):
        M = [row[:] for row in M]
        n = len(M)
        m = len(M[0]) - 1
        for i in range(n):
            max_row = i
            for k in range(i+1, n):
                if abs(M[k][i]) > abs(M[max_row][i]):
                    max_row = k
            if abs(M[max_row][i]) < 1e-12:
                continue
            M[i], M[max_row] = M[max_row], M[i]
            for k in range(i+1, n):
                if abs(M[i][i]) < 1e-12:
                    continue
                f = M[k][i] / M[i][i]
                for j in range(i, m+1):
                    M[k][j] -= f * M[i][j]
        
        pivots = []
        for i in range(n):
            for j in range(m):
                if abs(M[i][j]) > 1e-10:
                    pivots.append(j)
                    break
        free_vars = [j for j in range(m) if j not in pivots]
        if not free_vars:
            return []
        
        base = []
        for free in free_vars:
            vec = [0.0]*m
            vec[free] = 1.0
            for i in reversed(range(len(pivots))):
                if i < len(M):
                    linhai = M[i]
                    s = 0.0
                    for j in free_vars:
                        if j < len(linhai):
                            s += linhai[j]*vec[j]
                    if pivots[i] < len(linhai) and abs(linhai[pivots[i]]) > 1e-10:
                        vec[pivots[i]] = -s/linhai[pivots[i]]
            base.append(vec)
        return base

    @staticmethod
    def gauss_jordan(m):
        n_rows = len(m)
        n_cols = len(m[0])
        a = [row[:] for row in m]
        min_dim = min(n_rows, n_cols)
        row_used = [False]*n_rows
        for i in range(min_dim):
            max_row = None
            for r in range(i, n_rows):
                if abs(a[r][i]) > 1e-12 and not row_used[r]:
                    max_row = r
                    break
            if max_row is None:
                continue
            if max_row != i:
                a[i], a[max_row] = a[max_row], a[i]
            row_used[i] = True
            for j in range(n_rows):
                if (i != j):
                    if abs(a[i][i]) < 1e-12:
                        continue
                    p = a[j][i] / a[i][i]
                    for k in range(n_cols):
                        a[j][k] -= a[i][k] * p
        for i in range(min_dim):
            if abs(a[i][i]) > 1e-12:
                div = a[i][i]
                for k in range(n_cols):
                    a[i][k] /= div
        return a

    def nucleo(self, matriz=None):
        matriz_base = self.matriz if matriz is None else matriz
        A = np.array(matriz_base)
        if np.iscomplexobj(A):
            A = A.real
        else:
            A = A.astype(float)
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            return []
        n_linhas, n_colunas = A.shape
        M = np.hstack([A, np.zeros((n_linhas, 1))])
        return self.eliminacao_gauss(M.tolist())

    def imagem(self):
        A = np.array(self.matriz, dtype=float)
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            return []
        Aj = self.gauss_jordan(A.T.tolist())
        Aj = np.array(Aj).T
        base = []
        for j in range(Aj.shape[1]):
            col = Aj[:, j]
            if np.any(np.abs(col) > 1e-10):
                base.append([int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in col])
        return base

    @staticmethod
    def polinomio_caracteristico(matriz):
        matriz = np.array(matriz)
        n = matriz.shape[0]
        assert matriz.shape[1] == n
        coeficientes = np.array([1.])
        matriz_atual = np.array(matriz)
        for k in range(1, n + 1):
            coef = -np.trace(matriz_atual) / k
            coeficientes = np.append(coeficientes, coef)
            matriz_atual += np.diag(np.repeat(coef, n))
            matriz_atual = matriz @ matriz_atual
        return coeficientes

    def calcula_autovalores_autovetores(self, matriz, coeficientes):
        A = np.array(matriz, dtype=float)
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            return None, None
        n, m = A.shape
        if n != m:
            return None, None
        autovalores = self.calcula_autovalores(A, coeficientes)
        autovetores = self.calcula_autovetores(A, autovalores)
        return autovalores.real, autovetores

    def calcula_autovalores(self, matriz, coeficientes):
        autovalores = np.roots(coeficientes)
        autovalores = np.array([val.real if abs(val.imag) < 1e-10 else val for val in autovalores])
        return autovalores

    def calcula_autovetores(self, matriz, autovalores):
        A = np.array(matriz, dtype=float)
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            return np.array([])
        n, m = A.shape
        if n != m:
            return np.array([])
        I = np.eye(n)
        autovetores = []
        for val in autovalores:
            if np.iscomplex(val) and abs(val.imag) > 1e-10:
                autovetores.append(np.zeros(n))
                continue
            M = A - val * I
            base_nucleo = self.nucleo(M)
            vec = None
            if base_nucleo and self.norma_vetor(base_nucleo[0]) > 1e-12:
                vec = np.array(base_nucleo[0], dtype=float)
            else:
                u, s, vh = np.linalg.svd(M)
                vec = vh[-1, :]
            if vec is not None and self.norma_vetor(vec) > 1e-12:
                vec = vec / self.norma_vetor(vec)
                autovetores.append(vec.real)
            else:
                autovetores.append(np.zeros(n))
        return np.array(autovetores).T

    def formatar_vetor(self, v, casas_decimais=2):
        if isinstance(v, (list, np.ndarray)):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if hasattr(v, '__len__') and len(v) > 0 and not isinstance(v[0], (list, np.ndarray)):
                if any(isinstance(x, (list, np.ndarray)) for x in v):
                    return [self.formatar_vetor(x, casas_decimais) for x in v]
                return [self.float_para_fracao(x) for x in v]
            else:
                return [self.formatar_vetor(x, casas_decimais) for x in v]
        if isinstance(v, (float, np.floating)):
            return self.float_para_fracao(v)
        if v == -0.0 or v == 0.0 or v == -0:
            return 0
        return v

    @staticmethod
    def float_para_fracao(x, max_den=100):
        if isinstance(x, (float, np.floating)):
            if abs(x) < 1e-12 or x == -0.0:
                return 0
            if abs(x - round(x)) < 1e-8:
                return int(round(x))
            if abs(x) > 0 and abs(x) < 1:
                frac = Fraction(x).limit_denominator(100)
                if frac.denominator != 1:
                    num = int(round(x * 100))
                    den = 100
                    g = math.gcd(abs(num), den)
                    num //= g
                    den //= g
                    return f"{num}/{den}"
            frac = Fraction(x).limit_denominator(max_den)
            if abs(float(frac) - x) < 1e-8:
                if frac.denominator == 1:
                    return int(frac.numerator)
                return f"{frac.numerator}/{frac.denominator}"
            else:
                return round(x, 2)
        return x

    def norma_vetor(self, v):
        v = np.array(v, dtype=float)
        return sum(x**2 for x in v) ** 0.5


class Tela(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title("Calculadora - Álgebra Linear")
        self.root.geometry("1200x600")
        self.pack(fill="both", expand=True)

        # Tipo de entrada
        tk.Label(self, text='Tipo de operação:').place(x=50, y=10)
        self.tipo_var = tk.StringVar()
        self.combo_tipo = ttk.Combobox(self, textvariable=self.tipo_var, state="readonly")
        self.combo_tipo['values'] = (
            "Núcleo e Imagem de Transformação Linear",
            "Matriz em Bases Distintas", 
            "Autovalores e Autovetores",
            "Análise de Matriz Direta",
            "Base Ortonormal (Gram-Schmidt)",
            "Ângulo entre dois vetores"
        )
        self.combo_tipo.current(0)
        self.combo_tipo.place(x=160, y=10, width=250)
        self.combo_tipo.bind("<<ComboboxSelected>>", self.atualizar_campos)

        # Campos dinâmicos
        self.campos_frame = tk.Frame(self)
        self.campos_frame.place(x=30, y=40)
        
        # Caixa de texto com barra de rolagem para o resultado
        self.resultado_frame = tk.Frame(self)
        self.resultado_frame.place(x=500, y=10, width=680, height=570)
        self.resultado_scroll = tk.Scrollbar(self.resultado_frame, orient="vertical")
        self.resultado_text = tk.Text(self.resultado_frame, wrap="word", yscrollcommand=self.resultado_scroll.set, font=("Courier New", 16), state="disabled")
        self.resultado_scroll.config(command=self.resultado_text.yview)
        self.resultado_text.pack(side="left", fill="both", expand=True)
        self.resultado_scroll.pack(side="right", fill="y")
        
        self.botao_calcular = tk.Button(self, text='Calcular', command=self.calcular)
        self.botao_calcular.place(x=50, y=550, width=120, height=40)

        self.atualizar_campos()

    def limpar_campos(self):
        for widget in self.campos_frame.winfo_children():
            widget.destroy()

    def atualizar_campos(self, event=None):
        self.limpar_campos()
        tipo = self.tipo_var.get()
        
        if tipo == "Núcleo e Imagem de Transformação Linear":
            tk.Label(self.campos_frame, text='Transformação Linear (ex: T(x,y,z)=(x-y,x+y,3*x+y)):').grid(row=0, column=0, columnspan=2, sticky='w')
            self.input_transformacao = tk.Entry(self.campos_frame, width=60)
            self.input_transformacao.grid(row=1, column=0, columnspan=2, pady=5)
            
        elif tipo == "Matriz em Bases Distintas":
            tk.Label(self.campos_frame, text='Transformação Linear:').grid(row=0, column=0, columnspan=2, sticky='w')
            self.input_transformacao = tk.Entry(self.campos_frame, width=60)
            self.input_transformacao.grid(row=1, column=0, columnspan=2, pady=5)
            
            tk.Label(self.campos_frame, text='Base do Domínio (ex: 1,0,0;0,1,0;0,0,1):').grid(row=2, column=0, columnspan=2, sticky='w')
            self.input_base_dom = tk.Entry(self.campos_frame, width=60)
            self.input_base_dom.grid(row=3, column=0, columnspan=2, pady=5)
            
            tk.Label(self.campos_frame, text='Base do Contradomínio (mesmo formato):').grid(row=4, column=0, columnspan=2, sticky='w')
            self.input_base_cod = tk.Entry(self.campos_frame, width=60)
            self.input_base_cod.grid(row=5, column=0, columnspan=2, pady=5)
            
        elif tipo == "Autovalores e Autovetores":
            tk.Label(self.campos_frame, text='Transformação Linear (deve resultar em matriz quadrada):').grid(row=0, column=0, columnspan=2, sticky='w')
            self.input_transformacao = tk.Entry(self.campos_frame, width=60)
            self.input_transformacao.grid(row=1, column=0, columnspan=2, pady=5)
            
        elif tipo == "Análise de Matriz Direta":
            tk.Label(self.campos_frame, text='Nº Linhas:').grid(row=0, column=0, sticky='w')
            self.input_linhas = tk.Entry(self.campos_frame, width=5)
            self.input_linhas.grid(row=0, column=1)
            tk.Label(self.campos_frame, text='Nº Colunas:').grid(row=0, column=2, sticky='w')
            self.input_colunas = tk.Entry(self.campos_frame, width=5)
            self.input_colunas.grid(row=0, column=3)
            tk.Button(self.campos_frame, text='Gerar matriz', command=self.gerar_matriz_inputs).grid(row=0, column=4, padx=10)
            self.matriz_inputs_frame = tk.Frame(self.campos_frame)
            self.matriz_inputs_frame.grid(row=1, column=0, columnspan=5, pady=10)
            self.matriz_inputs = []

        elif tipo == "Base Ortonormal (Gram-Schmidt)":
            tk.Label(self.campos_frame, text='Base (ex: 1,0,0;0,1,0;0,0,1):').grid(row=0, column=0, sticky='w')
            self.input_base_gram = tk.Entry(self.campos_frame, width=60)
            self.input_base_gram.grid(row=1, column=0, pady=5)
            self.fraction_var = tk.BooleanVar(value=True)
            self.checkbox_fraction = tk.Checkbutton(self.campos_frame, text="Exibir como fração", variable=self.fraction_var)
            self.checkbox_fraction.grid(row=2, column=0, sticky='w')
        elif tipo == "Ângulo entre dois vetores":
            tk.Label(self.campos_frame, text='Vetor 1 (ex: 1,2,3):').grid(row=0, column=0, sticky='w')
            self.input_vetor1 = tk.Entry(self.campos_frame, width=40)
            self.input_vetor1.grid(row=1, column=0, pady=5)
            tk.Label(self.campos_frame, text='Vetor 2 (ex: 4,5,6):').grid(row=2, column=0, sticky='w')
            self.input_vetor2 = tk.Entry(self.campos_frame, width=40)
            self.input_vetor2.grid(row=3, column=0, pady=5)

    def gerar_matriz_inputs(self):
        for widget in self.matriz_inputs_frame.winfo_children():
            widget.destroy()
        try:
            linhas = int(self.input_linhas.get())
            colunas = int(self.input_colunas.get())
            self.matriz_inputs = []
            for i in range(linhas):
                linha_inputs = []
                for j in range(colunas):
                    entry = tk.Entry(self.matriz_inputs_frame, width=8)
                    entry.grid(row=i, column=j, padx=2, pady=2)
                    linha_inputs.append(entry)
                self.matriz_inputs.append(linha_inputs)
        except Exception:
            self.matriz_inputs = []

    def mostrar_resultado(self, texto):
        self.resultado_text.config(state="normal")
        self.resultado_text.delete(1.0, tk.END)
        self.resultado_text.insert(tk.END, texto)
        self.resultado_text.config(state="disabled")

        # Atualizar a barra de rolagem para o final
        self.resultado_text.see(tk.END)

    def formatar_resultado_vetor(self, vetor):
        if isinstance(vetor, list):
            return f"({', '.join(str(v) for v in vetor)})"
        return str(vetor)

    def gram_schmidt_ortogonal(self, bases):
        ortogonais = []
        for i, u in enumerate(bases):
            v = np.array(u, dtype=float)
            for j in range(i):
                vj = ortogonais[j]
                proj = np.dot(v, vj) / np.dot(vj, vj) * vj
                v = v - proj
            ortogonais.append(v)
        return ortogonais

    def normalizar_base(self, base):
        # Normaliza cada vetor da base ortogonal: v/||v||
        base_normal = []
        for v in base:
            norma = sum(x**2 for x in v) ** (1/2)
            base_normal.append([x / norma for x in v])
        return base_normal

    def formatar_vetor_gram(self, v, fracao=True):
        if isinstance(v, (list, np.ndarray)):
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if hasattr(v, '__len__') and len(v) > 0 and not isinstance(v[0], (list, np.ndarray)):
                if fracao:
                    return [TransformacaoLinear.float_para_fracao(x) for x in v]
                else:
                    return [round(float(x), 4) for x in v]
            else:
                return [self.formatar_vetor_gram(x, fracao) for x in v]
        if isinstance(v, (float, np.floating)):
            return TransformacaoLinear.float_para_fracao(v) if fracao else round(float(v), 4)
        if v == -0.0 or v == 0.0 or v == -0:
            return 0
        return v

    def angulo_entre_vetores(self, v1, v2):
        # Calcula o ângulo em radianos entre dois vetores
        v1 = np.array(v1, dtype=float)
        v2 = np.array(v2, dtype=float)
        dot = np.dot(v1, v2) # Produto interno
        norma1 = np.linalg.norm(v1)
        norma2 = np.linalg.norm(v2)
        if norma1 == 0 or norma2 == 0:
            raise ValueError("Um dos vetores é nulo (norma zero). Não é possível calcular o ângulo.")
        cos_theta = dot / (norma1 * norma2)
        # Corrigir possíveis erros numéricos fora do domínio de arccos
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return np.arccos(cos_theta)

    def calcular(self):
        tipo = self.tipo_var.get()
        try:
            if tipo == "Núcleo e Imagem de Transformação Linear":
                transformacao = self.input_transformacao.get()
                if not transformacao:
                    raise ValueError("Digite uma transformação linear")
                
                incognitas, expressoes = TransformacaoLinear.parse_transformacao(transformacao)
                if not incognitas or not expressoes:
                    raise ValueError("Formato inválido da transformação. Use: T(x,y,z)=(x-y,x+y,3*x+y)")
                
                T = TransformacaoLinear(incognitas, expressoes)
                
                resultado = []
                resultado.append(f"Transformação Linear: T: R^{len(incognitas)} → R^{len(expressoes)}")
                resultado.append(f"Variáveis: {', '.join(incognitas)}")
                resultado.append(f"Expressões: {', '.join(expressoes)}")
                resultado.append("\nMatriz associada:")
                for i, row in enumerate(T.matriz):
                    resultado.append(f"  {T.formatar_vetor(row)}")
                
                # Calcular núcleo
                base_nucleo = T.nucleo()
                resultado.append(f"\nKer(T) - Base do núcleo:")
                if base_nucleo:
                    for i, v in enumerate(base_nucleo):
                        resultado.append(f"  v{i+1} = {T.formatar_vetor(v)}")
                else:
                    resultado.append("  {0} (núcleo trivial)")
                resultado.append(f"Dimensão do núcleo: {len(base_nucleo)}")
                
                # Calcular imagem
                base_imagem = T.imagem()
                resultado.append(f"\nIm(T) - Base da imagem:")
                if base_imagem:
                    for i, v in enumerate(base_imagem):
                        resultado.append(f"  u{i+1} = {T.formatar_vetor(v)}")
                resultado.append(f"Dimensão da imagem: {len(base_imagem)}")
                
                # Verificar teorema do núcleo-imagem
                dim_dominio = len(incognitas)
                dim_nucleo = len(base_nucleo)
                dim_imagem = len(base_imagem)
                resultado.append(f"\nTeorema do Núcleo-Imagem:")
                resultado.append(f"dim(Dom) = dim(Ker) + dim(Im)")
                resultado.append(f"{dim_dominio} = {dim_nucleo} + {dim_imagem} = {dim_nucleo + dim_imagem}")
                if dim_dominio == dim_nucleo + dim_imagem:
                    resultado.append("✓ Verificado!")
                else:
                    resultado.append("⚠ Erro nos cálculos!")
                
                self.mostrar_resultado('\n'.join(resultado))
                
            elif tipo == "Matriz em Bases Distintas":
                transformacao = self.input_transformacao.get()
                base_dom_str = self.input_base_dom.get()
                base_cod_str = self.input_base_cod.get()
                
                if not all([transformacao, base_dom_str, base_cod_str]):
                    raise ValueError("Preencha todos os campos")
                
                incognitas, expressoes = TransformacaoLinear.parse_transformacao(transformacao)
                if not incognitas or not expressoes:
                    raise ValueError("Formato inválido da transformação")
                
                T = TransformacaoLinear(incognitas, expressoes)
                
                # Parse das bases
                try:
                    vetores_dom = [list(map(float, v.split(','))) for v in base_dom_str.strip().split(';') if v.strip()]
                    vetores_cod = [list(map(float, v.split(','))) for v in base_cod_str.strip().split(';') if v.strip()]
                except:
                    raise ValueError("Formato inválido das bases. Use: 1,0,0;0,1,0;0,0,1")
                
                B_dom = Base(np.array(vetores_dom).T)
                B_cod = Base(np.array(vetores_cod).T)
                
                matriz_nova = T.matriz_em_bases(B_dom, B_cod, opcao_menu='2')
                
                resultado = []
                resultado.append("Transformação Linear:")
                resultado.append(f"T: R^{len(incognitas)} → R^{len(expressoes)}")
                resultado.append(f"T({','.join(incognitas)}) = ({','.join(expressoes)})")
                
                resultado.append("\nMatriz na base canônica:")
                for row in T.matriz:
                    resultado.append(f"  {T.formatar_vetor(row)}")
                
                resultado.append(f"\nBase do domínio (colunas):")
                for i, col in enumerate(B_dom.vetores.T):
                    resultado.append(f"  b{i+1} = {T.formatar_vetor(col)}")
                
                resultado.append(f"\nBase do contradomínio (colunas):")
                for i, col in enumerate(B_cod.vetores.T):
                    resultado.append(f"  c{i+1} = {T.formatar_vetor(col)}")
                
                resultado.append("\nMatriz da transformação nas novas bases:")
                for row in matriz_nova:
                    resultado.append(f"  {T.formatar_vetor(row)}")
                
                self.mostrar_resultado('\n'.join(resultado))
                
            elif tipo == "Autovalores e Autovetores":
                transformacao = self.input_transformacao.get()
                if not transformacao:
                    raise ValueError("Digite uma transformação linear")
                
                incognitas, expressoes = TransformacaoLinear.parse_transformacao(transformacao)
                if not incognitas or not expressoes:
                    raise ValueError("Formato inválido da transformação")
                
                T = TransformacaoLinear(incognitas, expressoes)
                
                if T.matriz.shape[0] != T.matriz.shape[1]:
                    raise ValueError("A matriz não é quadrada. Autovalores só existem para matrizes quadradas.")
                
                resultado = []
                resultado.append(f"Transformação Linear: T: R^{len(incognitas)} → R^{len(expressoes)}")
                resultado.append(f"T({','.join(incognitas)}) = ({','.join(expressoes)})")
                
                resultado.append("\nMatriz associada:")
                for row in T.matriz:
                    resultado.append(f"  {T.formatar_vetor(row)}")
                
                # Calcular polinômio característico
                coeficientes = T.polinomio_caracteristico(T.matriz)
                resultado.append("\nPolinômio característico:")
                grau = len(coeficientes) - 1
                x = sp.symbols('t')
                termos = []
                for i, coef in enumerate(coeficientes):
                    pot = grau - i
                    if coef == 0:
                        continue
                    termo = sp.Mul(sp.Rational(coef), x**pot)
                    termos.append(termo)
                poly_expr = sp.Add(*termos)
                resultado.append(sp.pretty(poly_expr))
                
                # Calcular autovalores e autovetores
                autovalores, autovetores = T.calcula_autovalores_autovetores(T.matriz, coeficientes)
                
                resultado.append("\nAutovalores e autovetores:")
                if autovalores is not None and autovetores is not None:
                    for i, val in enumerate(autovalores):
                        resultado.append(f"\nt{i+1} = {T.formatar_vetor(val)}")
                        autovetor = autovetores[:, i]
                        resultado.append(f"  Autovetor: {T.formatar_vetor(autovetor)}")
                else:
                    resultado.append("Não foi possível calcular autovalores/autovetores.")
                
                self.mostrar_resultado('\n'.join(resultado))
                
            elif tipo == "Análise de Matriz Direta":
                if not self.matriz_inputs:
                    raise ValueError("Gere a matriz primeiro")
                
                matriz = []
                for linha_inputs in self.matriz_inputs:
                    linha = []
                    for entry in linha_inputs:
                        val_str = entry.get().strip()
                        if not val_str:
                            val_str = "0"
                        try:
                            linha.append(float(val_str))
                        except:
                            linha.append(0.0)
                    matriz.append(linha)
                
                A = np.array(matriz)
                
                # Criar uma transformação linear fictícia para usar os métodos
                incognitas = [f'x{i+1}' for i in range(A.shape[1])]
                T = TransformacaoLinear(incognitas, ['0'] * A.shape[0])
                T.matriz = A
                
                resultado = []
                resultado.append(f"Matriz {A.shape[0]}x{A.shape[1]}:")
                for row in A:
                    resultado.append(f"  {T.formatar_vetor(row)}")
                
                # Núcleo
                base_nucleo = T.nucleo()
                resultado.append(f"\nNúcleo da matriz:")
                if base_nucleo:
                    for i, v in enumerate(base_nucleo):
                        resultado.append(f"  v{i+1} = {T.formatar_vetor(v)}")
                else:
                    resultado.append("  {0} (núcleo trivial)")
                resultado.append(f"Dimensão do núcleo: {len(base_nucleo)}")
                
                # Imagem
                base_imagem = T.imagem()
                resultado.append(f"\nEspaço coluna (imagem):")
                if base_imagem:
                    for i, v in enumerate(base_imagem):
                        resultado.append(f"  u{i+1} = {T.formatar_vetor(v)}")
                resultado.append(f"Dimensão do espaço coluna: {len(base_imagem)}")
                
                # Se for quadrada, calcular autovalores
                if A.shape[0] == A.shape[1]:
                    try:
                        coeficientes = T.polinomio_caracteristico(A)
                        autovalores, autovetores = T.calcula_autovalores_autovetores(A, coeficientes)
                        
                        resultado.append(f"\nAutovalores e autovetores:")
                        if autovalores is not None and autovetores is not None:
                            for i, val in enumerate(autovalores):
                                resultado.append(f"\nt{i+1} = {T.formatar_vetor(val)}")
                                autovetor = autovetores[:, i]
                                resultado.append(f"  Autovetor: {T.formatar_vetor(autovetor)}")
                    except Exception as e:
                        resultado.append(f"\nErro ao calcular autovalores: {e}")
            
            elif tipo == "Base Ortonormal (Gram-Schmidt)":
                base_str = self.input_base_gram.get()
                if not base_str:
                    raise ValueError("Digite a base no formato: 1,0,0;0,1,0;0,0,1")
                try:
                    import sympy as sp
                    # Salvar as strings originais para exibição simbólica
                    base_strings = [
                        [x.strip() for x in v.split(',')] 
                        for v in base_str.strip().split(';') if v.strip()
                    ]
                    # Transformar em float para cálculo
                    base = [
                        [float(sp.sympify(x)) for x in v] 
                        for v in base_strings
                    ]
                except Exception as e:
                    raise ValueError(f"Formato inválido da base. Use: 1,0,0;0,1,0;0,0,1. Erro: {e}")
                ortogonais = self.gram_schmidt_ortogonal(base)
                ortonormais = self.normalizar_base(ortogonais)
                usar_fracao = self.fraction_var.get() if hasattr(self, 'fraction_var') else True
                resultado = []
                resultado.append("Base original:")
                for i, v in enumerate(base_strings):
                    exprs = [sp.sympify(x) for x in v]
                    resultado.append(f"  b{i+1} = ({', '.join([sp.pretty(e, use_unicode=True) for e in exprs])})")
                resultado.append("\nBase ortogonal (Gram-Schmidt):")
                for i, v in enumerate(ortogonais):
                    resultado.append(f"  v{i+1} = {self.formatar_vetor_gram(v, usar_fracao)}")
                resultado.append("\nBase ortonormal:")
                for i, v in enumerate(ortonormais):
                    resultado.append(f"  u{i+1} = {self.formatar_vetor_gram(v, usar_fracao)}")
                self.mostrar_resultado('\n'.join(resultado))
            elif tipo == "Ângulo entre dois vetores":
                v1_str = self.input_vetor1.get()
                v2_str = self.input_vetor2.get()
                if not v1_str or not v2_str:
                    raise ValueError("Digite ambos os vetores no formato: 1,2,3")
                try:
                    v1 = [float(x.strip()) for x in v1_str.split(',') if x.strip()]
                    v2 = [float(x.strip()) for x in v2_str.split(',') if x.strip()]
                except Exception as e:
                    raise ValueError(f"Formato inválido dos vetores. Use: 1,2,3. Erro: {e}")
                if len(v1) != len(v2):
                    raise ValueError("Os vetores devem ter a mesma dimensão.")
                angulo = self.angulo_entre_vetores(v1, v2)
                resultado = []
                resultado.append(f"Vetor 1: {v1}")
                resultado.append(f"Vetor 2: {v2}")
                resultado.append(f"\nÂngulo entre os vetores: {angulo} radianos")
                resultado.append(f"(em graus: {np.degrees(angulo)})")
                self.mostrar_resultado('\n'.join(resultado))
        except Exception as e:
            self.mostrar_resultado(f'Erro: {e}')


def main():
    root = tk.Tk()
    app = Tela(root)
    root.mainloop()

if __name__ == "__main__":
    main()