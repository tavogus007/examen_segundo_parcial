#HACERLO CORRER EN EL ENVIRONMENT DE LA EXPO DE INF 272, EN LA CARPETA QUE ESTA 
#EN EL ESCRITORIO, LA QUE DICE SQL Y PYTHON


import pyodbc

def comparar_datos_base_datos(query1, query2):
    try:
        # Conexión a la primera base de datos
        connection1 = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-MA5RDMP;DATABASE=bdexpo_instituo2;UID=sa;PWD=123')
        cursor1 = connection1.cursor()

        # Conexión a la segunda base de datos
        connection2 = pyodbc.connect('DRIVER={SQL Server};SERVER=DESKTOP-MA5RDMP;DATABASE=inf354;UID=sa;PWD=123')
        cursor2 = connection2.cursor()

        # Ejecución de las consultas
        cursor1.execute(query1)
        cursor2.execute(query2)

        # Obtención de los resultados
        results1 = cursor1.fetchall()
        results2 = cursor2.fetchall()

        # Comparación de los resultados
        coincidencias = []
        for row1 in results1:
            for row2 in results2:
                # Aquí debes definir la lógica de comparación
                if row1[0] == row2[0]:  # Ejemplo de comparación simple por el primer campo
                    coincidencias.append((row1, row2))

        return coincidencias

    except Exception as ex:
        print("Error durante la comparación de datos: {}".format(ex))

    finally:
        # Cierre de las conexiones
        connection1.close()
        connection2.close()

# Ejemplo de uso
query1 = "SELECT * FROM Calificacion WHERE ptje > 80"
query2 = "SELECT * FROM Calificacion WHERE ptje > 80"
coincidencias = comparar_datos_base_datos(query1, query2)
print(coincidencias)