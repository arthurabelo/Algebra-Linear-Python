import numpy as np
import re
# Exemplo BASES                           T(x,y,z)=(x-y,x+y,3x+y)
# Exemplo de uma base para outra          T(x,y,z,w)=(x+y-z+2w,3y+5w,2x-w)
#     Base 1                              1,1,3,0;0,1,1,2;0,4,1,1;5,-1,0,1
#     Base 2                              2,0,0;0,3,1;1,0,-1
# Exemplo de AUTOVALOR E AUTOVETOR        T(x,y,z)=(x+y-z,2y+2z,-3x+3y+z)
# Exemplo de AUTOVALOR E AUTOVETOR        T(x,y,z,w)=(x+2y,3y,2z,5z+3w)

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
            print("Erro ao interpretar a transformação linear. Verifique o formato.")
            return [], []


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


    def matriz_em_bases(self, base_dom, base_cod, opcao_menu=None):
        P = base_dom.vetores
        Q = base_cod.vetores
        # Se for opção 3 (autovalores/autovetores), só permite matriz quadrada
        if opcao_menu == '3':
            if self.matriz.shape[0] != self.matriz.shape[1]:
                raise ValueError("A matriz associada não é quadrada. Autovalores e autovetores só existem para matrizes quadradas.")
        # Se for opção 2, verificar se as dimensões batem com domínio e contradomínio
        if opcao_menu == '2':
            if P.shape[0] != len(self.incognitas):
                raise ValueError(f"A base do domínio deve ter dimensão {len(self.incognitas)} (número de variáveis de entrada).")
            if Q.shape[0] != len(self.expressoes):
                raise ValueError(f"A base do contradomínio deve ter dimensão {len(self.expressoes)} (número de variáveis de saída).")
        # Permitir bases não quadradas, só exigir inversa se for quadrada
        if Q.shape[0] == Q.shape[1]:
            Q_inv = np.linalg.inv(Q)
        else:
            Q_inv = np.linalg.pinv(Q)
        if P.shape[0] == P.shape[1]:
            P_inv = np.linalg.inv(P)
        else:
            P_inv = np.linalg.pinv(P)
        return Q_inv @ self.matriz @ P_inv
    

    def eliminacao_gauss(self, M):
        M = [row[:] for row in M]  # Cópia para não alterar original
        n = len(M)
        m = len(M[0]) - 1  # número de variáveis
        # Eliminação de Gauss
        for i in range(n):
            # Busca pivô
            max_row = i
            for k in range(i+1, n):
                if abs(M[k][i]) > abs(M[max_row][i]):
                    max_row = k
            if abs(M[max_row][i]) < 1e-12:
                continue
            # Troca linhas
            M[i], M[max_row] = M[max_row], M[i]
            # Zera abaixo
            for k in range(i+1, n):
                if abs(M[i][i]) < 1e-12:
                    continue
                f = M[k][i] / M[i][i]
                for j in range(i, m+1):
                    M[k][j] -= f * M[i][j]
        # Substituição reversa para encontrar solução geral
        # Variáveis livres: colunas sem pivô
        pivots = []
        for i in range(n):
            for j in range(m):
                if abs(M[i][j]) > 1e-10:
                    pivots.append(j)
                    break
        free_vars = [j for j in range(m) if j not in pivots]
        if not free_vars:
            return []  # Núcleo trivial: só o vetor nulo
        base = []
        for free in free_vars:
            vec = [0.0]*m
            vec[free] = 1.0
            for i in reversed(range(len(pivots))):
                linhai = M[i]
                s = 0.0
                for j in free_vars:
                    s += linhai[j]*vec[j]
                if abs(linhai[pivots[i]]) > 1e-10:
                    vec[pivots[i]] = -s/linhai[pivots[i]]
            base.append(vec)
        return base


    @staticmethod
    def gauss_jordan(m):
        n_rows = len(m)
        n_cols = len(m[0])
        a = [row[:] for row in m]  # Cópia da matriz
        min_dim = min(n_rows, n_cols)
        row_used = [False]*n_rows
        for i in range(min_dim):
            # Busca pivô na coluna i
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
            # Zera os outros elementos na coluna i
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
        A = np.array(self.matriz if matriz is None else matriz, dtype=float)
        # Proteção para matriz vazia ou 1D
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            print("Matriz inválida para cálculo do núcleo.")
            return []
        n_linhas, n_colunas = A.shape
        M = np.hstack([A, np.zeros((n_linhas, 1))])
        return self.eliminacao_gauss(M.tolist())


    def imagem(self):
        A = np.array(self.matriz, dtype=float)
        # Proteção para matriz vazia ou 1D
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            print("Matriz inválida para cálculo da imagem.")
            return []
        # Aplica gauss-jordan nas colunas (redução por colunas)
        # Transpõe, reduz por linhas, depois transpõe de volta
        Aj = self.gauss_jordan(A.T.tolist())
        Aj = np.array(Aj).T
        # Seleciona as colunas não nulas como base
        base = []
        for j in range(Aj.shape[1]):
            col = Aj[:, j]
            if np.any(np.abs(col) > 1e-10):
                base.append([int(round(x)) if abs(x - round(x)) < 1e-10 else x for x in col])
        return base


    @staticmethod
    def polinomio_caracteristico(matriz):
        '''
        Método de Leverrier-Faddeev (https://metodosnumericos.com.br/polinomio-caracteristico-o-metodo-de-leverrier-fadeev)
        Este método calcula os coeficientes do polinômio característico de uma matriz quadrada A,
        sem a necessidade de calcular determinantes. O algoritmo constrói iterativamente matrizes auxiliares
        e utiliza traços (soma dos elementos da diagonal principal) para encontrar os coeficientes do polinômio característico:
        det(A - xI) = c0*x^n + c1*x^(n-1) + ... + cn*x^0, onde c0, c1, ..., cn são os coeficientes.
        '''
        matriz = np.array(matriz)
        n = matriz.shape[0]
        assert matriz.shape[1] == n # Verifica se a matriz é quadrada
        coeficientes = np.array([1.])  # Inicializa array dos coeficientes (começa com 1 para x^n)
        matriz_atual = np.array(matriz) # Matriz auxiliar para iteração
        for k in range(1, n + 1):
            coef = -np.trace(matriz_atual) / k # Calcula o coeficiente usando o oposto do traço da matriz auxiliar dividido pelo expoente k
            coeficientes = np.append(coeficientes, coef) # Adiciona o coeficiente ao array
            matriz_atual += np.diag(np.repeat(coef, n)) # Atualiza a matriz auxiliar adicionando o coeficiente como uma matriz diagonal
            matriz_atual = matriz @ matriz_atual  # Multiplica a matriz original pela matriz auxiliar
        return coeficientes  # Retorna os coeficientes do polinômio característico na ordem do maior para o menor expoente
        

    def calcula_autovalores(self, matriz, coeficientes):
        autovalores = np.roots(coeficientes)
        # Remove pequenas partes imaginárias de autovalores reais
        autovalores = np.array([val.real if abs(val.imag) < 1e-10 else val for val in autovalores])
        return autovalores


    def calcula_autovetores(self, matriz, autovalores):
        """
        Calcula autovetores reais para cada autovalor fornecido.
        """
        A = np.array(matriz, dtype=float)
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            print("Matriz inválida para cálculo de autovetores.")
            return np.array([])
        n, m = A.shape
        if n != m:
            print("A matriz não é quadrada, autovetores não estão definidos.")
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
                print(f"Aviso: Não foi possível encontrar autovetor para autovalor {val}.")
                autovetores.append(np.zeros(n))
        return np.array(autovetores).T

    def calcula_autovalores_autovetores(self, matriz, coeficientes):
        """
        Calcula autovalores e autovetores reais para matrizes quadradas de qualquer tamanho.
        """
        A = np.array(matriz, dtype=float)
        # Proteção para matriz vazia ou 1D
        if len(A.shape) != 2 or A.shape[0] == 0 or A.shape[1] == 0:
            print("Matriz inválida para cálculo de autovalores/autovetores.")
            return None, None
        n, m = A.shape
        if n != m:
            print("A matriz não é quadrada, autovetores/autovalores não estão definidos.")
            return None, None
        autovalores = self.calcula_autovalores(A, coeficientes)
        autovetores = self.calcula_autovetores(A, autovalores)
        return autovalores.real, autovetores


    def imprimir_transformacao(self):
        print(f"T: R^{len(self.incognitas)} → R^{len(self.expressoes)}")
        print("Variáveis:", ', '.join(self.incognitas))
        print("Expressões:", ', '.join(self.expressoes))
        print("Matriz associada:")
        for row in self.matriz:
            print(self.formatar_vetor(row))


    def float_para_fracao(self, x, max_den=100):
        # Converte float para fração, se possível, senão arredonda
        from fractions import Fraction
        if isinstance(x, (float, np.floating)):
            if abs(x) < 1e-12 or x == -0.0:
                return int(0) * 0
            # Se for inteiro após arredondar, retorna como int
            if abs(x - round(x)) < 1e-8:
                return int(round(x))
            # Se for decimal entre 0 e 1 (excluindo 0), mostra como fração denominador 100
            if abs(x) > 0 and abs(x) < 1:
                frac = Fraction(x).limit_denominator(100)
                # Força denominador 100 se não for inteiro
                if frac.denominator != 1:
                    num = int(round(x * 100))
                    den = 100
                    # Simplifica se possível
                    from math import gcd
                    g = gcd(abs(num), den)
                    num //= g
                    den //= g
                    return f"{num}/{den}"
            # Caso geral: aproxima como fração
            frac = Fraction(x).limit_denominator(max_den)
            if abs(float(frac) - x) < 1e-8:
                if frac.denominator == 1:
                    return int(frac.numerator)
                return f"{frac.numerator}/{frac.denominator}"
            else:
                return int(round(x, 2))
        return x

    def formatar_vetor(self, v, casas_decimais=2):
        # Se for lista ou array, formata cada elemento
        if isinstance(v, (list, np.ndarray)):
            # Se for array numpy, converte para lista
            if isinstance(v, np.ndarray):
                v = v.tolist()
            # Se for lista de números, converte cada elemento
            if hasattr(v, '__len__') and len(v) > 0 and not isinstance(v[0], (list, np.ndarray)):
                # Se for matriz (lista de listas), formata cada linha
                if any(isinstance(x, (list, np.ndarray)) for x in v):
                    return [self.formatar_vetor(x, casas_decimais) for x in v]
                return [self.float_para_fracao(x) for x in v]
            else:
                return [self.formatar_vetor(x, casas_decimais) for x in v]
        # Se for float numpy
        if isinstance(v, (float, np.floating)):
            return self.float_para_fracao(v)
        # Se for -0.0, retorna 0
        if v == -0.0 or v == 0.0 or v == -0:
            return 0
        return v


    def norma_vetor(self, v):
        """
        Calcula a norma euclidiana de um vetor (lista ou array).
        """
        v = np.array(v, dtype=float)
        return sum(x**2 for x in v) ** 0.5


def menu():
    print("Menu:")
    print("1. Calcular base do núcleo e base da imagem da transformação linear")
    print("2. Calcular matriz da transformação em bases distintas")
    print("3. Calcular autovalores e autovetores da transformação linear")
    print("4. Alterar transformação linear")
    print("0. Sair")
    return input("Escolha uma opção: ")


def main():
    transformacao = input("Digite a transformação linear (ex: T(x,y,z)=(x-z,y+x,x,-3z)): ")
    incognitas, expressoes = TransformacaoLinear.parse_transformacao(transformacao)
    T = TransformacaoLinear(incognitas, expressoes)
    T.imprimir_transformacao()
    while True:
        op = menu()
        if op == '1':
            base_nucleo = T.nucleo()
            print("Ker(T) (base do núcleo):")
            for v in base_nucleo:
                print(T.formatar_vetor(v))
            print("Dimensão do núcleo:", len(base_nucleo))
            base_imagem = T.imagem()
            print("Im(T) (base da imagem):")
            for v in base_imagem:
                print(T.formatar_vetor(v))
            print("Dimensão da imagem:", len(base_imagem))
        elif op == '2':
            # Antes de tudo, verifica se a matriz associada é válida
            if not hasattr(T, 'matriz') or not isinstance(T.matriz, np.ndarray) or T.matriz.size == 0 or len(T.matriz.shape) != 2:
                print("Transformação linear inválida. Defina uma transformação válida antes de calcular em outras bases.")
                continue
            print(
                "Digite a base do domínio (cada vetor separado por ponto e vírgula, ex: 1,0,0;0,1,0;0,0,1):"
            )
            base_dom = input()
            if not base_dom.strip():
                print("Base do domínio não pode ser vazia.")
                continue
            try:
                vetores_dom = [
                    list(map(float, v.split(',')))
                    for v in base_dom.strip().split(';') if v.strip() != ''
                ]
            except Exception:
                print("Base do domínio inválida. Tente novamente.")
                continue
            print("Digite a base do contradomínio (mesmo formato):")
            base_cod = input()
            if not base_cod.strip():
                print("Base do contradomínio não pode ser vazia.")
                continue
            try:
                vetores_cod = [
                    list(map(float, v.split(',')))
                    for v in base_cod.strip().split(';') if v.strip() != ''
                ]
            except Exception:
                print("Base do contradomínio inválida. Tente novamente.")
                continue
            print("Matriz da base do domínio (colunas):\n", np.array(vetores_dom).T)
            print("Matriz da base do contradomínio (colunas):\n", np.array(vetores_cod).T)
            B_dom = Base(np.array(vetores_dom).T)
            B_cod = Base(np.array(vetores_cod).T)
            try:
                matriz_nova = T.matriz_em_bases(B_dom, B_cod, opcao_menu='2')
            except Exception as e:
                print(f"Erro ao calcular matriz nas novas bases: {e}")
                continue
            print("Matriz da transformação nas novas bases:")
            for row in matriz_nova:
                print(T.formatar_vetor(row))
        elif op == '3':
            # Verifica se a matriz é quadrada e válida antes de calcular autovalores/autovetores
            if (not hasattr(T, 'matriz') or not isinstance(T.matriz, np.ndarray)
                or T.matriz.size == 0 or len(T.matriz.shape) != 2):
                print("Transformação linear inválida. Defina uma transformação válida antes de calcular autovalores e autovetores.")
                continue
            if T.matriz.shape[0] != T.matriz.shape[1]:
                print("A matriz associada não é quadrada. Autovalores e autovetores só existem para matrizes quadradas.")
                continue
            coeficientes = T.polinomio_caracteristico(T.matriz)
            print("Polinômio característico:")
            termos = []
            grau = len(coeficientes) - 1
            for i, c in enumerate(coeficientes):
                pot = grau - i
                coef_str = f"{T.formatar_vetor(c)}"
                if coef_str == "1" and pot != 0:
                    coef_str = ""
                elif coef_str == "-1" and pot != 0:
                    coef_str = "-"
                if pot == 0:
                    termo = f"{coef_str}"
                elif pot == 1:
                    termo = f"{coef_str}x"
                else:
                    termo = f"{coef_str}x^{pot}"
                termos.append(termo)
            polinomio = termos[0]
            for termo in termos[1:]:
                if termo.startswith('-'):
                    polinomio += f"{termo}"
                else:
                    polinomio += f"+{termo}"
            print(polinomio)

            autovalores, autovetores = T.calcula_autovalores_autovetores(T.matriz, coeficientes)
            print("Autovalores e autovetores:")
            if autovalores is not None and autovetores is not None:
                for i, val in enumerate(autovalores):
                    print(f"λ = {T.formatar_vetor(val)}")
                    autovetor = autovetores[:, i]
                    print(f"  Autovetor: {T.formatar_vetor(autovetor)}")
            else:
                print("Não foi possível calcular autovalores/autovetores.")
        elif op == '4':
            transformacao = input(
                "Digite a nova transformação linear (ex: T(x,y,z)=(x-z,y+x,x,-3z)): "
            )
            incognitas, expressoes = TransformacaoLinear.parse_transformacao(
                transformacao)
            if not incognitas or not expressoes:
                print("Transformação inválida. A transformação anterior será mantida.")
                continue
            T = TransformacaoLinear(incognitas, expressoes)
            T.imprimir_transformacao()
        elif op == '0':
            print("Saindo...")
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()