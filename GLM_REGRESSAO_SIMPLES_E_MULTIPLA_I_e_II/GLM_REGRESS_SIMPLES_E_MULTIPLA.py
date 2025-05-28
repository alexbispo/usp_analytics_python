# UNIVERSIDADE DE SÃO PAULO
# INTRODUÇÃO AO PYTHON E MACHINE LEARNING
# GLM - REGRESSÃO SIMPLES E MÚLTIPLA
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários
    
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
from scipy.stats import pearsonr # correlações de Pearson
from sklearn.preprocessing import LabelEncoder # transformação de dados


# In[ ]:
#############################################################################
#                          REGRESSÃO LINEAR SIMPLES                         #
#                  EXEMPLO 01 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################
    
df = pd.read_csv('tempodist.csv', delimiter=',')
df

#Características das variáveis do dataset
df.info()

#Estatísticas univariadas
df.describe()


# In[ ]: Gráfico de dispersão

#Regressão linear que melhor se adequa às obeservações: função 'sns.lmplot'

plt.figure(figsize=(20,10))
sns.lmplot(data=df, x='distancia', y='tempo', ci=False)
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=14)
plt.show


# In[ ]: Gráfico de dispersão

#Regressão linear que melhor se adequa às obeservações: função 'sns.regplot'

plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=False, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Estimação do modelo de regressão linear simples

#Estimação do modelo
modelo = sm.OLS.from_formula('tempo ~ distancia', df).fit()

#Observação dos parâmetros resultantes da estimação
modelo.summary()


# In[ ]: Salvando fitted values (variável yhat) 
# e residuals (variável erro) no dataset

df['yhat'] = modelo.fittedvalues
df['erro'] = modelo.resid
df


# In[ ]: Gráfico didático para visualizar o conceito de R²

y = df['tempo']
yhat = df['yhat']
x = df['distancia']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot([x[i],x[i]], [yhat[i],y[i]],'--', color='#2ecc71')
    plt.plot([x[i],x[i]], [yhat[i],mean[i]], ':', color='#9b59b6')
    plt.plot(x, y, 'o', color='#2c3e50')
    plt.axhline(y = y.mean(), color = '#bdc3c7', linestyle = '-')
    plt.plot(x,yhat, color='#34495e')
    plt.title('R2: ' + str(round(modelo.rsquared,4)))
    plt.xlabel("Distância")
    plt.ylabel("Tempo")
    plt.legend(['Erro = Y - Yhat', 'Yhat - Ymédio'], fontsize=10)
plt.show()


# In[ ]: Cálculo manual do R²

R2 = ((df['yhat']-
       df['tempo'].mean())**2).sum()/(((df['yhat']-
                                        df['tempo'].mean())**2).sum()+
                                        (df['erro']**2).sum())

round(R2,4)


# In[ ]: Coeficiente de ajuste (R²) é a correlação ao quadrado

#Correlação de Pearson
df[['tempo','distancia']].corr()

#R²
(df[['tempo','distancia']].corr())**2

#R² de maneira direta
modelo.rsquared


# In[ ]: Modelo auxiliar para mostrar R² igual a 100% (para fins didáticos)

#Estimação do modelo com yhat como variável dependente,
#resultará em uma modelo com R² igual a 100%
modelo_auxiliar = sm.OLS.from_formula('yhat ~ distancia', df).fit()

#Parâmetros resultantes da estimação
modelo_auxiliar.summary()


# In[ ]:Gráfico mostrando o perfect fit

plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='yhat', ci=False, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Voltando ao nosso modelo original


#Plotando o intervalo de confiança de 90%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=90, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#%%
#Plotando o intervalo de confiança de 95%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=95, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#%%
#Plotando o intervalo de confiança de 99%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=99, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#%%
#Plotando o intervalo de confiança de 99,999%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=99.999, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Calculando os intervalos de confiança

#Nível de significância de 10% / Nível de confiança de 90%
modelo.conf_int(alpha=0.1)

#Nível de significância de 5% / Nível de confiança de 95%
modelo.conf_int(alpha=0.05)

#Nível de significância de 1% / Nível de confiança de 99%
modelo.conf_int(alpha=0.01)

#Nível de significância de 0,001% / Nível de confiança de 99,999%
modelo.conf_int(alpha=0.00001)


# In[ ]: Fazendo predições em modelos OLS
#Ex.: Qual seria o tempo gasto, em média, para percorrer a distância de 25km?

modelo.predict(pd.DataFrame({'distancia':[25]}))

#Cálculo manual - mesmo valor encontrado
5.8784 + 1.4189*(25)


# In[ ]: Nova modelagem para o mesmo exemplo, com novo dataset que
#contém replicações

#Quantas replicações de cada linha você quer? -> função 'np.repeat'
df_replicado = pd.DataFrame(np.repeat(df.values, 3, axis=0))
df_replicado.columns = df.columns
df_replicado


# In[ ]: Estimação do modelo com valores replicados

modelo_replicado = sm.OLS.from_formula('tempo ~ distancia',
                                       df_replicado).fit()

#Parâmetros do modelo
modelo_replicado.summary()


# In[ ]: Calculando os novos intervalos de confiança

#Nível de significância de 5% / Nível de confiança de 95%
modelo_replicado.conf_int(alpha=0.05)


# In[ ]: Plotando o novo gráfico com intervalo de confiança de 95%
#Note o estreitamento da amplitude dos intervalos de confiança!

plt.figure(figsize=(20,10))
sns.regplot(data=df_replicado, x='distancia', y='tempo', ci=95, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: PROCEDIMENTO ERRADO: ELIMINAR O INTERCEPTO QUANDO ESTE NÃO SE MOSTRAR
#ESTATISTICAMENTE SIGNIFICANTE

modelo_errado = sm.OLS.from_formula('tempo ~ 0 + distancia', df).fit()

#Parâmetros do modelo
modelo_errado.summary()


# In[ ]: Comparando os parâmetros do modelo inicial (objeto 'modelo')
#com o 'modelo_errado' pela função 'summary_col' do pacote
#'statsmodels.iolib.summary2'

summary_col([modelo, modelo_errado])

#Outro modo mais completo também pela função 'summary_col'
summary_col([modelo, modelo_errado],
            model_names=["MODELO INICIAL","MODELO ERRADO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })


# In[ ]: Gráfico didático para visualizar o viés decorrente de se eliminar
# erroneamente o intercepto em modelos regressivos

x = df['distancia']
y = df['tempo']

yhat = df['yhat']
yhat_errado = modelo_errado.fittedvalues

plt.plot(x, y, 'o', color='dimgray')
plt.plot(x, yhat, color='limegreen')
plt.plot(x, yhat_errado, color='red')
plt.xlabel("Distância")
plt.ylabel("Tempo")
plt.legend(['Valores Observados','Fitted Values - Modelo Coreto',
            'Fitted Values - Modelo Errado'], fontsize=9)
plt.show()


# In[ ]:
#############################################################################
#                         REGRESSÃO LINEAR MÚLTIPLA                         #
#                EXEMPLO 02 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_paises = pd.read_csv('paises.csv', delimiter=',', encoding="utf-8")
df_paises

#Características das variáveis do dataset
df_paises.info()

#Estatísticas univariadas
df_paises.describe()


# In[ ]: Gráfico 3D com scatter

import plotly.io as pio
pio.renderers.default = 'browser'

trace = go.Scatter3d(
    x=df_paises['horas'], 
    y=df_paises['idade'], 
    z=df_paises['cpi'], 
    mode='markers',
    marker={
        'size': 5,
        'opacity': 0.8,
    },
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)
plot_figure.update_layout(scene = dict(
                        xaxis_title='horas',
                        yaxis_title='idade',
                        zaxis_title='cpi'))
plot_figure.show()


# In[ ]: Matriz de correlações

corr = df_paises.drop(columns=['pais']).corr()
corr

labels = ['cpi', 'idade', 'horas']

fig, ax = plt.subplots(figsize=(8, 6))

cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

fig.colorbar(cax)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

for (i, j), val in np.ndenumerate(corr):
    ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

plt.title('Matriz de Correlação')

plt.show()

#Palettes de cores
#sns.color_palette("viridis", as_cmap=True)
#sns.color_palette("magma", as_cmap=True)
#sns.color_palette("inferno", as_cmap=True)
#sns.color_palette("Blues", as_cmap=True)
#sns.color_palette("Greens", as_cmap=True)
#sns.color_palette("Reds", as_cmap=True)


# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_paises, diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Estimando um modelo múltiplo com as variáveis do dataset 'paises'

#Estimando a regressão múltipla
modelo_paises = sm.OLS.from_formula("cpi ~ idade + horas", df_paises).fit()

#Parâmetros do modelo
modelo_paises.summary()

#Parâmetros dos modelo com intervalos de confiança
#Nível de significância de 5% / Nível de confiança de 95%
modelo_paises.conf_int(alpha=0.05)


# In[ ]: Salvando os fitted values na base de dados

df_paises['cpifit'] = modelo_paises.fittedvalues
df_paises


# In[ ]: Gráfico 3D com scatter e fitted values resultantes do modelo

trace = go.Scatter3d(
    x=df_paises['horas'], 
    y=df_paises['idade'], 
    z=df_paises['cpi'], 
    mode='markers',
    marker={
        'size': 5,
        'opacity': 0.8,
    },
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
    xaxis_title='X AXIS TITLE',
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)
plot_figure.add_trace(go.Mesh3d(
                    x=df_paises['horas'], 
                    y=df_paises['idade'], 
                    z=df_paises['cpifit'], 
                    opacity=0.5,
                    color='pink'
                  ))
plot_figure.update_layout(scene = dict(
                        xaxis_title='horas',
                        yaxis_title='idade',
                        zaxis_title='cpi'))
plot_figure.show()


# In[ ]:
#############################################################################
#         REGRESSÃO COM UMA VARIÁVEL EXPLICATIVA (X) QUALITATIVA            #
#             EXEMPLO 03 - CARREGAMENTO DA BASE DE DADOS                    #
#############################################################################

df_corrupcao = pd.read_csv('corrupcao.csv',delimiter=',',encoding='utf-8')
df_corrupcao

#Características das variáveis do dataset
df_corrupcao.info()

#Estatísticas univariadas
df_corrupcao.describe()

# Estatísticas univariadas por região
df_corrupcao.groupby('regiao').describe()

#Tabela de frequências da variável 'regiao'
#Função 'value_counts' do pacote 'pandas' sem e com o argumento 'normalize'
#para gerar, respectivamente, as contagens e os percentuais
contagem = df_corrupcao['regiao'].value_counts(dropna=False)
percent = df_corrupcao['regiao'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=False)


# In[ ]: Conversão dos dados de 'regiao' para dados numéricos, a fim de
#se mostrar a estimação de modelo com o problema da ponderação arbitrária

label_encoder = LabelEncoder()
df_corrupcao['regiao_numerico'] = label_encoder.fit_transform(df_corrupcao['regiao'])
df_corrupcao['regiao_numerico'] = df_corrupcao['regiao_numerico'] + 1
df_corrupcao.head(10)

#A nova variável 'regiao_numerico' é quantitativa (ERRO!), fato que
#caracteriza a ponderação arbitrária!
df_corrupcao['regiao_numerico'].info()
df_corrupcao.describe()


# In[ ]: Modelando com a variável preditora numérica, resultando na
#estimação ERRADA dos parâmetros
#PONDERAÇÃO ARBITRÁRIA!
modelo_corrupcao_errado = sm.OLS.from_formula("cpi ~ regiao_numerico",
                                              df_corrupcao).fit()

#Parâmetros do modelo
modelo_corrupcao_errado.summary()

#Calculando os intervalos de confiança com nível de significância de 5%
modelo_corrupcao_errado.conf_int(alpha=0.05)


# In[ ]: Plotando os fitted values do modelo_corrupcao_errado considerando,
#PROPOSITALMENTE, a ponderação arbitrária, ou seja, assumindo que as regiões
#representam valores numéricos (América do Sul = 1; Ásia = 2; EUA e Canadá = 3;
#Europa = 4; Oceania = 5).

ax =sns.lmplot(
    data=df_corrupcao,
    x="regiao_numerico", y="cpi",
    height=10
)
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']) + " " +
                str(point['y']))
plt.title('Resultado da Ponderação Arbitrária', fontsize=16)
plt.xlabel('Região', fontsize=14)
plt.ylabel('Corruption Perception Index', fontsize=14)
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca()) 


# In[ ]: Dummizando a variável 'regiao'. O código abaixo automaticamente fará: 
# a)o estabelecimento de dummies que representarão cada uma das regiões do dataset; 
# b)removerá a variável original a partir da qual houve a dummização; 
# c)estabelecerá como categoria de referência a primeira categoria, ou seja,
# a categoria 'America_do_sul' por meio do argumento 'drop_first=True'.

df_corrupcao_dummies = pd.get_dummies(df_corrupcao, columns=['regiao'],
                                      drop_first=True)

df_corrupcao_dummies.head(10)

#A variável 'regiao' está inicialmente definida como 'object' no dataset
df_corrupcao.info()
#O procedimento atual também poderia ter sido realizado em uma variável
#dos tipos 'category' ou 'string'. Para fins de exemplo, podemos transformar a
#variável 'regiao' para 'category' ou 'string' e comandar o código anterior:
df_corrupcao['regiao'] = df_corrupcao['regiao'].astype("category")
df_corrupcao.info()
df_corrupcao['regiao'] = df_corrupcao['regiao'].astype("string")
df_corrupcao.info()


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

modelo_corrupcao_dummies = sm.OLS.from_formula("cpi ~ regiao_Asia + \
                                              regiao_EUA_e_Canada + \
                                              regiao_Europa + \
                                              regiao_Oceania",
                                              df_corrupcao_dummies).fit()

#Parâmetros do modelo
modelo_corrupcao_dummies.summary()

#Outro método de estimação (sugestão de uso para muitas dummies no dataset)
# Definição da fórmula utilizada no modelo
lista_colunas = list(df_corrupcao_dummies.drop(columns=['cpi','pais','regiao_numerico']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "cpi ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_corrupcao_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_corrupcao_dummies).fit()

#Parâmetros do modelo
modelo_corrupcao_dummies.summary()


# In[ ]: Plotando o modelo_corrupcao_dummies de forma interpolada

#Fitted values do 'modelo_corrupcao_dummies' no dataset 'df_corrupcao_dummies'
df_corrupcao_dummies['fitted'] = modelo_corrupcao_dummies.fittedvalues
df_corrupcao_dummies.head()


# In[ ]: Gráfico propriamente dito

from scipy import interpolate

plt.figure(figsize=(10,10))

df2 = df_corrupcao_dummies[['regiao_numerico','fitted']].groupby(['regiao_numerico']).median().reset_index()
x = df2['regiao_numerico']
y = df2['fitted']

tck = interpolate.splrep(x, y, k=2)
xnew = np.arange(1,5,0.1) 
ynew = interpolate.splev(xnew, tck, der=0)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']) + " " + str(point['y']))

plt.scatter(df_corrupcao_dummies['regiao_numerico'], df_corrupcao_dummies['cpi'])
plt.scatter(df_corrupcao_dummies['regiao_numerico'], df_corrupcao_dummies['fitted'])
plt.plot(xnew, ynew)
plt.title('Ajuste Não Linear do Modelo com Variáveis Dummy', fontsize=16)
plt.xlabel('Região', fontsize=14)
plt.ylabel('Corruption Perception Index', fontsize=14)
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca())


# In[ ]:
#############################################################################
#            REGRESSÃO NÃO LINEAR E TRANSFORMAÇÃO DE BOX-COX                #
#              EXEMPLO 04 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_bebes = pd.read_csv('bebes.csv', delimiter=',')
df_bebes

#Características das variáveis do dataset
df_bebes.info()

#Estatísticas univariadas
df_bebes.describe()


# In[ ]: Gráfico de dispersão

plt.figure(figsize=(10,10))
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='black',
                s=100, label='Valores Reais')
plt.title('Dispersão dos dados', fontsize=17)
plt.xlabel('Idade em semanas', fontsize=16)
plt.ylabel('Comprimento em cm', fontsize=16)
plt.legend(loc='lower right', fontsize=16)
plt.show()


# In[ ]: Estimação de um modelo OLS linear
modelo_linear = sm.OLS.from_formula('comprimento ~ idade', df_bebes).fit()

#Observar os parâmetros resultantes da estimação
modelo_linear.summary()


# In[ ]: Gráfico de dispersão com ajustes (fits) linear e não linear

plt.figure(figsize=(10,10))
sns.regplot(x="idade", y="comprimento", data=df_bebes, order=2,
            color='darkviolet', ci=False, scatter=False, label='Ajuste Não Linear')
plt.plot(df_bebes['idade'], modelo_linear.fittedvalues, color='darkorange',
         label='OLS Linear')
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='black',
                s=100, label='Valores Reais')
plt.title('Dispersão dos dados e ajustes linear e não linear', fontsize=17)
plt.xlabel('Idade em semanas', fontsize=16)
plt.ylabel('Comprimento em cm', fontsize=16)
plt.legend(loc='lower right', fontsize=16)
plt.show()


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro
#shapiro(modelo_linear.resid)

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import shapiro_francia
shapiro_francia(modelo_linear.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_linear.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)

print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Histograma dos resíduos do modelo OLS linear

plt.figure(figsize=(10,10))
sns.histplot(data=modelo_linear.resid, kde=True, bins=30, color = 'darkorange')
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#x é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
x, lmbda = boxcox(df_bebes['comprimento'])

#Inserindo a variável transformada ('bc_comprimento') no dataset
#para a estimação de um novo modelo
df_bebes['bc_comprimento'] = x

df_bebes

#Apenas para fins de comparação e comprovação do cálculo de x
df_bebes['bc_comprimento2'] = ((df_bebes['comprimento']**lmbda)-1)/lmbda

df_bebes

del df_bebes['bc_comprimento2']


# In[ ]: Estimando um novo modelo OLS com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_comprimento ~ idade', df_bebes).fit()

#Parâmetros do modelo
modelo_bc.summary()


# In[ ]: Comparando os parâmetros do 'modelo_linear' com os do 'modelo_bc'
#CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!

summary_col([modelo_linear, modelo_bc])

#Outro modo mais completo também pela função 'summary_col'
summary_col([modelo_linear, modelo_bc],
            model_names=["MODELO LINEAR","MODELO BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

#Repare que há um salto na qualidade do ajuste para o modelo não linear (R²)

pd.DataFrame({'R² OLS':[round(modelo_linear.rsquared,4)],
              'R² Box-Cox':[round(modelo_bc.rsquared,4)]})


# In[ ]: Verificando a normalidade dos resíduos do 'modelo_bc'

# Teste de Shapiro-Francia
shapiro_francia(modelo_bc.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_bc.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Histograma dos resíduos do modelo_bc

plt.figure(figsize=(10,10))
sns.histplot(data=modelo_bc.resid, kde=True, bins=30, color='darkviolet')
plt.xlabel('Resíduos', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Fazendo predições com os modelos OLS linear e Box-Cox
#Qual é o comprimento esperado de um bebê com 52 semanas de vida?

#Modelo OLS Linear:
modelo_linear.predict(pd.DataFrame({'idade':[52]}))

#Modelo Não Linear (Box-Cox):
modelo_bc.predict(pd.DataFrame({'idade':[52]}))

#Não podemos nos esquecer de fazer o cálculo inverso para a obtenção do fitted
#value de Y (variável 'comprimento')
(54251.109775 * lmbda + 1) ** (1 / lmbda)


# In[ ]: Salvando os fitted values dos dois modelos (modelo_linear e modelo_bc)
#no dataset 'bebes'

df_bebes['yhat_linear'] = modelo_linear.fittedvalues
df_bebes['yhat_modelo_bc'] = (modelo_bc.fittedvalues * lmbda + 1) ** (1 / lmbda)
df_bebes


# In[ ]: Gráfico de dispersão com ajustes dos modelos OLS linear e Box-Cox

plt.figure(figsize=(10,10))
sns.regplot(x="idade", y="yhat_modelo_bc", data=df_bebes, order=lmbda,
            color='darkviolet', ci=False, scatter=False, label='Box-Cox')
plt.scatter(x="idade", y="yhat_modelo_bc", data=df_bebes, alpha=0.5,
            s=60, color='darkviolet', label='Fitted Values Box-Cox')
sns.regplot(x="idade", y="yhat_linear", data=df_bebes,
            color='darkorange', ci=False, scatter=False, label='OLS Linear')
plt.scatter(x="idade", y="yhat_linear", data=df_bebes, alpha=0.5,
            s=60, color='darkorange', label='Fitted Values OLS Linear')
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='black',
                s=100, label='Valores Reais')
plt.title('Dispersão dos dados e ajustes dos modelos OLS linear e Box-Cox',
          fontsize=17)
plt.xlabel('Idade em semanas', fontsize=16)
plt.ylabel('Comprimento em cm', fontsize=16)
plt.legend(loc='lower right', fontsize=16)
plt.show()


# In[ ]: Ajustes dos modelos
#valores previstos (fitted values) X valores reais

from scipy.optimize import curve_fit

def objective(x, a, b, c, d, e):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + e

xdata = df_bebes['comprimento']
ydata_linear = df_bebes['yhat_linear']
ydata_bc = df_bebes['yhat_modelo_bc']

plt.figure(figsize=(10,10))

popt, _ = curve_fit(objective, xdata, ydata_linear)
a, b, c, d, e = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e)
plt.plot(x_line, y_line, '--', color='darkorange', linewidth=3)

popt, _ = curve_fit(objective, xdata, ydata_bc)
a, b, c, d, e = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e)
plt.plot(x_line, y_line, '--', color='darkviolet', linewidth=3)

plt.plot(xdata,xdata, color='gray', linestyle='-')
plt.scatter(xdata, ydata_linear, alpha=0.5, s=100, color='darkorange')
plt.scatter(xdata, ydata_bc, alpha=0.5, s=100, color='darkviolet')
plt.title('Dispersão e Fitted Values dos Modelos Linear e Box-Cox',
          fontsize=17)
plt.xlabel('Valores Reais de Comprimento', fontsize=16)
plt.ylabel('Fitted Values', fontsize=16)
plt.legend(['OLS Linear','Box-Cox','45º graus'], fontsize=17)
plt.show()


# In[ ]:
#############################################################################
#                        REGRESSÃO NÃO LINEAR MÚLTIPLA                      #
#                  EXEMPLO 05 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################

df_empresas = pd.read_csv('empresas.csv', delimiter=',')
df_empresas

#Características das variáveis do dataset
df_empresas.info()

#Estatísticas univariadas
df_empresas.describe()

print(df_empresas.head())

# In[ ]: Matriz de correlações

#Maneira simples pela função 'corr'

numeric_df_empresas = df_empresas.select_dtypes(include=['number'])

# Calcular a correlação
corr = numeric_df_empresas.corr()

#Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg

corr2 = pg.rcorr(df_empresas, method='pearson',
                 upper='pval', decimals=4,
                 pval_stars={0.01: '***',
                             0.05: '**',
                             0.10: '*'})
corr2


# In[ ]: Mapa de calor da matriz de correlações

plt.figure(figsize=(15, 10))
ax = plt.gca()

# Plota o heatmap
cax = ax.imshow(corr, interpolation='nearest', cmap=plt.cm.viridis)

# Adiciona a barra de cores
plt.colorbar(cax)

# Define os rótulos dos eixos
ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.index)))
ax.set_xticklabels(corr.columns, rotation=90, fontsize=12)
ax.set_yticklabels(corr.index, fontsize=12)

# Adiciona as anotações na matriz
for i in range(len(corr.columns)):
    for j in range(len(corr.index)):
        ax.text(j, i, f'{corr.iat[i, j]:.2f}', ha='center', va='center', color='white', fontsize=15)

# Ajusta os limites dos eixos
ax.set_xlim(-0.5, len(corr.columns) - 0.5)
ax.set_ylim(len(corr.index) - 0.5, -0.5)

# Mostra o plot
plt.show()

# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_empresas, diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Estimando a Regressão Múltipla
modelo_empresas = sm.OLS.from_formula('retorno ~ disclosure +\
                                      endividamento + ativos +\
                                          liquidez', df_empresas).fit()

# Parâmetros do modelo
modelo_empresas.summary()

#Note que o parâmetro da variável 'endividamento' não é estatisticamente
#significante ao nível de significância de 5% (nível de confiança de 95%).

# Cálculo do R² ajustado (slide 15 da apostila)
r2_ajust = 1-((len(df_empresas.index)-1)/(len(df_empresas.index)-\
                                          modelo_empresas.params.count()))*\
    (1-modelo_empresas.rsquared)
r2_ajust # modo direto: modelo_empresas.rsquared_adj


# In[ ]: Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'statstests.process'
# Autores do pacote: Helder Prado Santos e Luiz Paulo Fávero
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_empresas = stepwise(modelo_empresas, pvalue_limit=0.05)


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro
#shapiro(modelo_step_empresas.resid)

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import shapiro_francia
shapiro_francia(modelo_step_empresas.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_empresas.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Plotando os resíduos do 'modelo_step_empresas' e acrescentando
#uma curva normal teórica para comparação entre as distribuições

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_empresas.resid, fit=norm, kde=True, bins=20,
             color='goldenrod')
plt.xlabel('Resíduos do Modelo Linear', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#'x' é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
x, lmbda = boxcox(df_empresas['retorno'])

print("Lambda: ",lmbda)


# In[ ]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_empresas['bc_retorno'] = x
df_empresas

#Verificação do cálculo, apenas para fins didáticos
df_empresas['bc_retorno2'] = ((df_empresas['retorno'])**(lmbda) - 1) / (lmbda)
df_empresas

del df_empresas['bc_retorno2']


# In[ ]: Estimando um novo modelo múltiplo com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_retorno ~ disclosure +\
                                endividamento + ativos +\
                                    liquidez', df_empresas).fit()

# Parâmetros do modelo
modelo_bc.summary()


# In[ ]: Aplicando o procedimento Stepwise no 'modelo_bc"

modelo_step_empresas_bc = stepwise(modelo_bc, pvalue_limit=0.05)

#Note que a variável 'disclosure' retorna ao modelo na forma funcional
#não linear!


# In[ ]: Verificando a normalidade dos resíduos do 'modelo_step_empresas_bc'

# Teste de Shapiro-Francia
shapiro_francia(modelo_step_empresas_bc.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_empresas_bc.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Plotando os novos resíduos do 'modelo_step_empresas_bc'

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_empresas_bc.resid, fit=norm, kde=True, bins=20,
             color='red')
plt.xlabel('Resíduos do Modelo Box-Cox', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Resumo dos dois modelos obtidos pelo procedimento Stepwise
#(linear e com Box-Cox)

summary_col([modelo_step_empresas, modelo_step_empresas_bc],
            model_names=["STEPWISE","STEPWISE BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

#CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!


# In[ ]: Fazendo predições com o 'modelo_step_empresas_bc'
# Qual é o valor do retorno, em média, para 'disclosure' igual a 50,
#'liquidez' igual a 14 e 'ativos' igual a 4000, ceteris paribus?

modelo_step_empresas_bc.predict(pd.DataFrame({'const':[1],
                                              'disclosure':[50],
                                              'ativos':[4000],
                                              'liquidez':[14]}))


# In[ ]: Não podemos nos esquecer de fazer o cálculo para a obtenção do
#fitted value de Y (variável 'retorno')

(3.702016 * lmbda + 1) ** (1 / lmbda)


# In[ ]: Salvando os fitted values de 'modelo_step_empresas' e
#'modelo_step_empresas_bc'

df_empresas['yhat_step_empresas'] = modelo_step_empresas.fittedvalues
df_empresas['yhat_step_empresas_bc'] = (modelo_step_empresas_bc.fittedvalues
                                        * lmbda + 1) ** (1 / lmbda)

#Visualizando os dois fitted values no dataset
#modelos 'modelo_step_empresas e modelo_step_empresas_bc
df_empresas[['empresa','retorno','yhat_step_empresas','yhat_step_empresas_bc']]


# In[ ]: Ajustes dos modelos: valores previstos (fitted values) X valores reais

from scipy.optimize import curve_fit

def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

xdata = df_empresas['retorno']
ydata_linear = df_empresas['yhat_step_empresas']
ydata_bc = df_empresas['yhat_step_empresas_bc']

plt.figure(figsize=(10,10))

popt, _ = curve_fit(objective, xdata, ydata_linear)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='goldenrod', linewidth=3)

popt, _ = curve_fit(objective, xdata, ydata_bc)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='red', linewidth=3)

plt.plot(xdata,xdata, color='gray', linestyle='-')
plt.scatter(xdata,ydata_linear, alpha=0.5, s=100, color='goldenrod')
plt.scatter(xdata,ydata_bc, alpha=0.5, s=100, color='red')
plt.title('Dispersão e Fitted Values dos Modelos Linear e Box-Cox',
          fontsize=17)
plt.xlabel('Valores Reais de Retorno', fontsize=16)
plt.ylabel('Fitted Values', fontsize=16)
plt.legend(['Stepwise','Stepwise com Box-Cox','45º graus'], fontsize=17)
plt.show()


# In[ ]:
#############################################################################
#      DIAGNÓSTICO DE HETEROCEDASTICIDADE EM MODELOS DE REGRESSÃO           #
#              EXEMPLO 06 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################
    
df_saeb_rend = pd.read_csv("saeb_rend.csv", delimiter=',')
df_saeb_rend

#Características das variáveis do dataset
df_saeb_rend.info()

#Estatísticas univariadas
df_saeb_rend.describe()


# In[ ]: Tabela de frequências absolutas das variáveis 'uf' e rede'

df_saeb_rend['uf'].value_counts()
df_saeb_rend['rede'].value_counts()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com linear fit

plt.figure(figsize=(15,10))
sns.regplot(x='rendimento', y='saeb', data=df_saeb_rend, marker='o',
            fit_reg=True, color='green', ci=False,
            scatter_kws={"color":'gold', 'alpha':0.5, 's':150})
plt.title('Gráfico de Dispersão com Ajuste Linear', fontsize=20)
plt.xlabel('rendimento', fontsize=17)
plt.ylabel('saeb', fontsize=17)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'rede' escolar

plt.figure(figsize=(15,10))
sns.scatterplot(x='rendimento', y='saeb', data=df_saeb_rend,
                hue='rede', alpha=0.5, s=120, palette = 'viridis')
plt.title('Gráfico de Dispersão com Ajuste Linear', fontsize=20)
plt.xlabel('rendimento', fontsize=17)
plt.ylabel('saeb', fontsize=17)
plt.legend(loc='upper left', fontsize=17)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'rede' escolar e linear fits - Gráfico pela função 'lmplot' do 'seaborn' com
#estratificação de 'rede' pelo argumento 'hue'

plt.figure(figsize=(15,10))
sns.lmplot(x='rendimento', y='saeb', data=df_saeb_rend,
           hue='rede', ci=None, palette='viridis')
plt.title('Gráfico de Dispersão com Ajuste Linear por Rede', fontsize=14)
plt.xlabel('rendimento', fontsize=12)
plt.ylabel('saeb', fontsize=12)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'rede' escolar e linear fits - Gráfico pela função 'regplot' do 'seaborn'

plt.figure(figsize=(15,10))
df1 = df_saeb_rend[df_saeb_rend['rede'] == 'Municipal']
df2 = df_saeb_rend[df_saeb_rend['rede'] == 'Estadual']
df3 = df_saeb_rend[df_saeb_rend['rede'] == 'Federal']
sns.regplot(x='rendimento', y='saeb', data=df1, ci=False, marker='o',
            scatter_kws={"color":'darkorange', 'alpha':0.3, 's':150},
            label='Municipal')
sns.regplot(x='rendimento', y='saeb', data=df2, ci=False, marker='o',
            scatter_kws={"color":'darkviolet', 'alpha':0.3, 's':150},
            label='Estadual')
sns.regplot(x='rendimento', y='saeb', data=df3, ci=False, marker='o',
            scatter_kws={"color":'darkgreen', 'alpha':0.8, 's':150},
            label='Federal')
plt.title('Gráfico de Dispersão com Ajuste Linear por Rede', fontsize=20)
plt.xlabel('rendimento', fontsize=17)
plt.ylabel('saeb', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Estimação do modelo de regressão e diagnóstico de heterocedasticidade

# Estimando o modelo
modelo_saeb = sm.OLS.from_formula('saeb ~ rendimento', df_saeb_rend).fit()

# Parâmetros do modelo
modelo_saeb.summary()


# In[ ]: Adicionando fitted values e resíduos do 'modelo_saeb'
#no dataset 'df_saeb_rend'

df_saeb_rend['fitted'] = modelo_saeb.fittedvalues
df_saeb_rend['residuos'] = modelo_saeb.resid
df_saeb_rend


# In[ ]: Gráfico que relaciona resíduos e fitted values do
#'modelo_saeb'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted', y='residuos', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=20)
plt.xlabel('Fitted Values do Modelo', fontsize=17)
plt.ylabel('Resíduos do Modelo', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Plotando os resíduos do 'modelo_saeb' e acrescentando
#uma curva normal teórica para comparação entre as distribuições
#Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_saeb.resid, fit=norm, kde=True, bins=15,
             color='red')
sns.kdeplot(data=modelo_saeb.resid, multiple="stack", alpha=0.4,
            color='red')
plt.xlabel('Resíduos do Modelo', fontsize=16)
plt.ylabel('Densidade', fontsize=16)
plt.show()


# In[ ]: Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

from scipy import stats

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value


# In[ ]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_saeb)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')


# In[ ]: Dummizando a variável 'uf'

df_saeb_rend_dummies = pd.get_dummies(df_saeb_rend, columns=['uf'],
                                      drop_first=True)

df_saeb_rend_dummies


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_saeb_rend_dummies.drop(columns=['municipio',
                                                        'codigo',
                                                        'escola',
                                                        'rede',
                                                        'saeb',
                                                        'fitted',
                                                        'residuos']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "saeb ~ " + formula_dummies_modelo

modelo_saeb_dummies_uf = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_saeb_rend_dummies).fit()

#Parâmetros do modelo
modelo_saeb_dummies_uf.summary()

#Estimação do modelo por meio do procedimento Stepwise
from statstests.process import stepwise
modelo_saeb_dummies_uf_step = stepwise(modelo_saeb_dummies_uf, pvalue_limit=0.05)


# In[ ]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_saeb_dummies_uf_step'

breusch_pagan_test(modelo_saeb_dummies_uf_step)

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb_dummies_uf_step) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')


# In[ ]: Adicionando fitted values e resíduos do 'modelo_saeb_dummies_uf_step'
#no dataset 'df_saeb_rend'

df_saeb_rend['fitted_step'] = modelo_saeb_dummies_uf_step.fittedvalues
df_saeb_rend['residuos_step'] = modelo_saeb_dummies_uf_step.resid
df_saeb_rend


# In[ ]: Gráfico que relaciona resíduos e fitted values do
#'modelo_saeb_dummies_uf_step'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'dodgerblue', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=20)
plt.xlabel('Fitted Values do Modelo Stepwise com Dummies', fontsize=17)
plt.ylabel('Resíduos do Modelo Stepwise com Dummies', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Plotando os resíduos do 'modelo_saeb_dummies_uf_step' e acrescentando
#uma curva normal teórica para comparação entre as distribuições
#Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_saeb_dummies_uf_step.resid, fit=norm, kde=True, bins=15,
             color='dodgerblue')
sns.kdeplot(data=modelo_saeb_dummies_uf_step.resid, multiple="stack", alpha=0.4,
            color='dodgerblue')
plt.xlabel('Resíduos do Modelo', fontsize=16)
plt.ylabel('Densidade', fontsize=16)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'uf' e linear fits - Gráfico pela função 'lmplot' do pacote 'seaborn', com
#estratificação de 'uf' pelo argumento 'hue'

plt.figure(figsize=(15,10))
sns.lmplot(x='rendimento', y='saeb', data=df_saeb_rend,
           hue='uf', ci=None, palette='viridis')
plt.title('Gráfico de Dispersão com Ajuste Linear por UF', fontsize=14)
plt.xlabel('rendimento', fontsize=12)
plt.ylabel('saeb', fontsize=12)
plt.show()


################################### FIM ######################################