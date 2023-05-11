''' ejemplo de multiprocesamiento a la hora de aplicar funciones a los DataFrame'''

# Muchas funciones  del paquete Scikit-learn funcionan utilizando multiprocesadores, gracias al
# paquete joblib. Dicho paquete requiere que todas las funciones se ejecuten en múltiples
# procesadores para que sean importadas, y no puede aceptarlas si se definen sobre la 
# marcha (deben ser seleccionables). Una posible solución es guardar la función en un 
# archivo en el disco, como hemos realizado para este ejemplo.


from multiprocessing import Pool
import numpy as np
import pandas as pd


def square(number):
    ''' funcion para aplicar a un elemento del Dataframe o a cualquier objeto'''
    return number**2


def apply_df(dframe):
    ''' funcion para enlazar cada trozo del data frame con la funcion a aplicar'''
    return dframe.apply(square)


# este if es necesario pq los subprocesos no saben que son subprocesos e intentan
# ejecutar el script principal de forma recursiva.
if __name__ == '__main__':

    iris = pd.read_csv(
        'resources\datasets-uci-iris.csv',
        names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])

    # extraer las culumnas a aplicar la funcion
    df = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

    # dividir el DataFrame en 5 trozos
    df_split = np.array_split(df, 5)

    # crear un pool de 4 proccesos (optimo para in Intel i5)
    with Pool(4) as p:

        # aplicar la funcion a cada trozo sobre el pool que creamos
        results = p.map(apply_df, df_split)

    # unir los trozos e imprimirlos
    print(pd.concat(list(results)).head())
