#!/usr/bin/env python
# coding: utf-8

# # Theoretische Grundlagen der Wärme- und Feuchtspeicherung und des -transportes
# 
# Wenn eine Koppelung zwischen Wärme- und Feuchtetransport in einer Berechnung besteht spricht man allgemein von einer
# hygrothermischen Simulation. Diese Koppelung bringt den Vorteil, dass der Einfluss des Feuchtegehaltes auf die
# Wärmeleitfähigkeit berücksichtigt werden kann. Die Differentialgleichungen, welche für solche Berechnungen miteinander
# gekoppelt werden müssen, werden im folgenden vorgestellt, mit dem Ziel ein Verständnis für die unterliegenden Grundlagen
# und Konzepte zu vermitteln.
# 
# ## Allgemeine Transportgleichungen
# 
# ### Wärmespeicherung und -transport
# 
# Bei Betrachtung der Wärmeleitung ergibt sich aus Gleichung {eq}`delta_V_q_eq`
# 
# $$ \rho c \frac{\partial T}{\partial t} = -\nabla \cdot q + h $$(dT_eq)
# 
# Dabei beschreibt $h$ (J/(m³s)) eine Wärmequelle. Diese kann ausgelöst sein durch mechanisches Heizen, chemische
# Reaktionen etc.
# 
# Die Wärmestromdichte q (J/(m²s) ist definiert als
# 
# $$ q = q_{diff} + q_{conv}$$ (q_diff_conv_eq)
# 
# wobei $q_{diff}$ den diffusiven Anteil und $q_{conv}$ den durch Konvektion tranportierten Anteil der Wärmestromdichte
# beschreibt. Bei Vernachlässigung konvektiver Phänomene können wir $q_{conv} = 0$ setzen.
# 
# Für den diffusiven eindimensionalen Fall ist $q$ definiert als
# 
# $$ q = q_{diff} = \lambda \cdot \frac{T - (T + \Delta T)}{\Delta x}$$(q_delta_x_eq)
# $$ q = -\lambda \cdot \frac{\Delta T}{\Delta x}$$(q_delta_x_eq)
# 
# ```{figure} img/Wärmebrücke/q_steady_state_wall.png
# ---
# height: 250px
# name: q_steady_state_wall
# ---
# Stationärer Wärmefluss durch ein finites Wandstück. {cite}`hagentoftIntroductionBuildingPhysics2001`
# ```
# 
# Gleichung {eq}`q_delta_x_eq` kann bei Grenzwertbetrachtung $\lim \limits_{\Delta x \to 0}$ übergeführt werden in den
# Differentialoperator
# 
# $$ q = -\lambda \cdot \frac{d T}{d x}$$(q_dx_eq)
# 
# Mehrdimensional verallgemeinern lässt sich Gleichung {eq}`q_dx_eq` mittels des Nabla-Operators zu
# 
# $$ q = -\lambda \cdot \nabla T$$(q_nabla_eq)
# 
# ```{note}
# $\lambda$ kann eine vektorielle Größe sein wodurch richtungsabhängigen Wärmeleitfähigkeiten berücksichtigt werden können.
# Falls also $\lambda_{x} \neq \lambda_{y} \neq \lambda_{z}$.
# ```
# 
# Gleichung {eq}`q_nabla_eq` eingesetzt in Gleichung {eq}`dT_eq` führt auf
# 
# $$ \frac{\partial T}{\partial t} = \frac {\lambda}{\rho c} \nabla \cdot (\nabla T) = \nabla^{2}T$$(dT_laplace_eq)
# 
# Dabei beschreibt $\nabla^{2}$ den Laplace-Operator:
# 
# $$ \nabla^{2} = \frac{\partial^{2}}{\partial x^{2}} + \frac{\partial^{2}}{\partial y^{2}} + \frac{\partial^{2}}{\partial
# z^{2}} $$
# 
# $$ \nabla^{2}T = \frac{\partial^{2}T_{x}}{\partial x^{2}} + \frac{\partial^{2}T_{y}}{\partial y^{2}} +
# \frac{\partial^{2}T_{z}}{\partial z^{2}} $$
# 
# Multiplikation der Wärmestromdichte mit der entsprechend durchströmten Fläche $A$ in jm² führt auf den Wärmestrom $Q$ in
# J/s.
# 
# ### Feuchtespeicherung und -transport
# 
# Analog zum Wärmetransport kann der Feuchtetransport durch Gleichung {eq}`w_eq` beschrieben werden:
# 
# $$ \frac{\partial w}{\partial t} = - \nabla \cdot g + m $$(w_eq)
# 
# Hier steht $\frac{\partial w}{\partial t}$ für die Veränderung des Wassergehaltes in kg/m³s, $g$ für die
# Feuchtestromdichte in kg/(m²s) und m für eine Feuchtequelle in kg/(m³s). Die Feuchtestromdichte setzt sich in porösen
# Medien zusammen aus der
# 
# - Dampfdiffusionsstromdichte $g_{v}$ und
# - Flüssigwasserstromdichte $g_{l}$.
# 
# $$ g = g_{v} + g_{l} $$ (g_eq)
# 
# Wenn wir vorübergehend den Flüssigwassertransport vernachlässigen $g_{l} = 0$, was bei entsprechenden Randbedingungen
# eine [valide Annahme](Randbedingungen_Diffusive) darstellt, setzt sich die Dampfdiffusionsstromdichte $g_{v}$, ähnlich
# wie der Wärmefluss {eq}`q_diff_conv_eq`, aus einem diffusiven und konvektiven Anteil zusammen
# 
# $$ g = g_{v} = g_{diff} + g_{conv} $$ (g_diff_conv_eq)
# 
# Bei Vernachlässigung konvektiver Phänomene können wir wiederum $g_{conv} = 0$ setzen. Den noch verbleibenden Anteil der
# Feuchtstromdichte können wir in porösen Medien über das Fick'sche Gesetz beschreiben.
# 
# ```{figure} img/Wärmebrücke/g_steady.png
# ---
# height: 250px
# name: g_steady
# ---
# Feuchtestromdichte durch ein finites poröses Wandstück. {cite}`hagentoftIntroductionBuildingPhysics2001`
# ```
# 
# Dabei wird die Diffusion in ruhender Luft in Abhängigkeit gesetzt zu dem Dampfdruckgradienten $\Delta v$ und einer
# Transportkonstante $\delta_{v}$.
# 
# $$ g = g_{diff} = \delta_{v} \cdot \frac{v - (v + \Delta v)}{\Delta x} $$
# 
# $$ g = -\delta_{v} \cdot \frac{\Delta v}{\Delta x} $$ (g_delta_eq)
# 
# $$ g = -\frac{D}{\mu} \cdot \frac{\Delta v}{\Delta x} $$ (g_delta_eq)
# 
# Hierbei steht D (m²/s) für den temperaturabhängigen Diffusionskoeffizienten ruhender Luft, wobei $T$ in Grad Celsius
# einzusetzen ist
# 
# $$ D = (22.2 + 0.14 \cdot T) \cdot 10^{-6}$$
# 
# und $\mu$ für einen Faktor den Widerstand zu beschreiben der zu Folge des porösen Mediums besteht. Dieser Widerstand
# entsteht einerseits durch
# 
# - Reduktion des Vorhandenen Raumes für Diffusion und andererseits
# - zu Folge der längeren Wege die ein Dampf-Teilchen zu Folge eines Porennetzwerkes zurücklegen muss.
# 
# Analog zum Wärmefluss kann Gleichung {eq}`g_delta_eq` bei Grenzwertbetrachtung $\lim \limits_{\Delta x \to 0}$
# übergeführt werden in den Differentialoperator
# 
# $$ g = -\delta_{v} \cdot \frac{d v}{\Delta x} $$ (g_dx_eq)
# 
# und mittels des Nabal-Operators mehrdimensional verallgemeinert werden zu
# 
# $$ g = - \delta_{v} \cdot \nabla v $$ (g_nabla_eq)
# 
# ## Numerische Lösung einer partiellen Differentialgleichung am Beispiel des Temperaturfeldes
# 
# ```{note}
# Die hier dargestellten Ausführungen sind ein kurzer Auszug aus {cite}`waltherBuildingPhysicsApplications2021`. Das Werk
# ermöglicht einen einfachen, anwendungsbezogenen Einstieg in die numerische Lösung komplexerer (z.B. zeitabhängig, ortsabhängig) bauphysikalischer Problemstellungen.
# 
# Die dargebrachten einfachen Methoden der Numerik, werden für das bessere Verständnis in einem simplen Python-Code implementiert und können über Binder Interaktiv auf der HTML-Seite ausprobiert werden.
# ```
# Zur numerischen Lösung partieller Differentialgleichungen müssen einerseits die Gleichungen und andererseits die Problemdomäne diskretisiert werden. Dafür gibt es unterschiedliche Ansätze:
# 
# - Finite Differenzen
# - Finite Elemente
# - Finite Volumen etc.
# 
# Hier wird nur ein kurzer Überblick über die Finite Differenzen Methode gebracht und direkt auf die Wärmeleitungsgleichung angewendet.
# 
# Da es sich bei der Wärmeleitungsgleichung um ein zeitabhängiges Problem handelt muss die Veränderung der Temperatur nach der Zeit $\frac{dT}{dt}$ anschließend numerisch auf-integriert werden. Ansätze dafür wären:
# 
# - Euler-Verfahren
# - Runge-Kutta-Verfahren
# - Adams-Bashforth-Verfahren etc.
# 
# Wir werden die algebraische Lösung der zeitlichen Ableitung mittels des Euler-Verfahrens integrieren um somit das Temperaturfeld zu lösen.
# 
# ```{note}
# Wahl des Diskretisierungs und Integrationsverfahrens ist abhängig von der Problemstellung.
# ```
# 
# ### Finite Differenzen Methode
# #### Diskretisierung der Geometrie
# 
# Da es numerisch nicht möglich ist kontinuierliche Probleme zu lösen muss als erster Schritt die Geometrie diskretisiert werden. Mittels der FDM (Finite Differenzen Metode) geschieht das auf die denkbar simpelste Weise. Die Problemdomände wird durch Knotenpunkte (Grid Points) diskretisiert wodurch ein Raster entsteht {numref}`wl_diskret`.
# 
# ```{figure} img/Wärmebrücke/wl_diskret.png
# ---
# height: 250px
# name: wl_diskret
# ---
# Diskretisierung der geometrischen Problemdomäne.
# ```
# 
# Jedem dieser Knotenpunkte sind die entsprechenden physikalischen Eigenschaften zugewiesen (Materialparameter, Temperaturen etc.) um alle Eingangswerte zu haben, um damit im nächsten Schritt die beschreibenden Differentialgleichungen für diese Knoten zu lösen und somit die orts- und zeitabhängige Veränderung in diesen darstellen zu können.
# 
# #### Diskretisierung der Differentialgleichung
# 
# Die Finite Differenzen Methode ermöglicht die Diskretisierung partieller Differentialgleichung und damit ihre Überführung in numerisch lösbare Formen. Wir betrachten für unser Beispiel einen eindimensionalen Fall (z.B einen Stab) mit konstanter Wärmeleitfähigkeit. Wenn wir $a$ definieren als $a = \frac {\lambda}{\rho c}$ sieht die Diffusionsgleichung für Wärmeleitung folgendermaßen aus:
# 
# $$ \frac{dT}{dt} = a \frac{dT}{dx^{2}}$$(temp_1d_eq)
# 
# Für ein diskretes Element $i$ führt die Darstellung mittels finiter Differenzen auf:
# 
# $$ \frac{T_{i}^{+} - T_{i}}{\Delta t} = a \frac{T_{i+1} -2 T_{i} + T_{i-1}}{\Delta x^{2}}$$(disk_temp_eq)
# 
# Wenn wir als Randbedingungen $T_{(x=0)} = 0$ °C und $T_{(x=L)} = 10$ °C {numref}`wl_diskret_RB`, Starttemperatur für das zu berechnende Feld $T_{x=1\ bis\ L-1} = 0$ °C und z.B. $t = 50$ Sekunden (Simulationszeitraum) wählen, können wir nun mittels der rechten Seite von Gleichung {eq}`disk_temp_eq` die Verändeurng der Temperatur für jeden Zeitschritt berechnen.
# 
# ```{figure} img/Wärmebrücke/wl_diskret_RB.png
# ---
# height: 200px
# name: wl_diskret_RB
# ---
# Äußersten Knoten für die Randbedingungen.
# ```
# 
# ### Zeitliche Integration - Euler-Verfahren
# 
# Nach jedem Zeitschritt muss mittels des Euler-Verfahrens aufintegriert werden. Betrachten wir die linke Seite von Gleichung {eq}`disk_temp_eq` können wir wiederholen, dass
# 
# $$ \frac{dT}{dt} \approx  \frac{T_{i}^{+} - T_{i}}{\Delta t} $$(time_disk_eq)
# 
# ist. Wir wissen weiters auch, dass die zeitliche Veränderung der Temperatur eine Funktion ist
# 
# $$ \frac{dT}{dt} = f $$(dt_f_eq)
# 
# Diese Funktion kennen wir {eq}`temp_1d_eq`. Wenn wir nun {eq}`dt_f_eq` in {eq}`time_disk_eq` einsetzen und umformen, erhalten wir folgenden Zusammenhang
# 
# $$ T_{i}^{+} = T_{i} + \Delta t f $$(dt_euler_eq)
# 
# und mittels unserer Definition von $f$ aus Gleichung {eq}`temp_1d_eq` und {eq}`disk_temp_eq` erhalten wir
# 
# $$ T_{i}^{+} = T_{i} + \Delta t a \frac{T_{i+1} -2 T_{i} + T_{i-1}}{\Delta x^{2}} $$(T_disk_eq)

# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Anzahl der Knoten (Diskretisierung)
resolution = 100

# T and T+ Vektoren erstellen und mit Startwert 0 °C initialisieren
T_plus = np.zeros(resolution)
T = np.zeros(resolution)

# Simulations-Zeit
sim_time = 50
dt = 0.01

#Geometrie und thermische Parameter
L = 0.1  # m
dx = L / resolution  #
lambda_mat = 50  # W/(mK)
rho = 7850  # kg/m³
c = 500  # J/(kgK)
alpha = lambda_mat / (rho * c)
Fo = alpha * dt / dx ** 2  # Fourier-Zahl


# In[3]:


# Stabilitätskriterium
if Fo > 0.5:
    print("stability issue")

# Zeit
t = 0
while t < sim_time:
    # Randbedingungen bleiben gleich
    T[0] = 0
    T[resolution - 1] = 10
    # Berechnung innerhalb der Domäne
    for i in range(1, resolution - 1):
        # Ergebnis für neuen Zeitschritt berechnen
        T_plus[i] = T[i] * (1 - 2 * Fo) + Fo * (T[i + 1] + T[i - 1])

    # Altes Temperaturfeld mit soeben berechnetem überschreiben
    T = T_plus
    # Zeitschritt dazuzählen
    t += dt
# plotting
x_pos = np.arange(0, L, dx)
plt.xlabel("x Position in m")
plt.ylabel("Temperatur in °C")
plt.grid()
plt.plot(x_pos, T_plus)
plt.show()


# Wenn wir ausreichend Zeitschritte rechnen und dabei die Randbedingungen nicht verändern konvergiert das Ergebnis gegen die stationäre Lösung des Problems. {numref}`T_one_Side`.
# 
# ```{figure} img/Wärmebrücke/T_one_Side.gif
# ---
# height: 250px
# name: T_one_Side
# ---
# Zeitlicher Darstellung der Entwicklung des Temperatufeldes in einem Stab mit den Randbedingungen $T_{(x=0)} = 0$ °C und $T_{(x=L)} = 10$ °C.
# ```

# Für einen beidseitig erhitzten Stab $T_{(x=0)} = 10$ °C und $T_{(x=L)} = 10$ °C ist die zeitliche Entwiklung des Temperaturfeldes in {numref}`T_both_sides` dargestellt.
# 
# ```{figure} img/Wärmebrücke/T_both_sides.gif
# ---
# height: 250px
# name: T_both_sides
# ---
# Zeitlicher Darstellung der Entwicklung des Temperatufeldes in einem Stab mit den Randbedingungen $T_{(x=0)} = 10$ °C und $T_{(x=L)} = 10$ °C. Nach einer ausreichenden Zeitdauer herrscht im gesamten Stab dieselbe Temperatur.
# ```

# ```{note}
# Auf Grund der zeitlichen Variation des Wetters sind bei hygrothermischen Simulationen die Randbedingungen auch eine Funktion der Zeit.
# ```

# (Randbedingungen_Diffusive)=
# 
# ## Randbedingungen
# 
# Der Wärmeübergangswiderstand $R_{s}$ in m²K/W setzt sich aus dem konvektiven $\alpha_{c}$ und und radiativen
# Wärmeübergangskoeffizienten $\alpha_{r}$ zusammen:
# 
# $$ R_{s} = \frac{1}{\alpha_{c} + \alpha_{r}}$$(R_s_eq)
# 
# ```{figure} img/Wärmebrücke/wärme_BT.png
# ---
# height: 350px
# name: wärme_BT
# ---
# 
# ```
# 
# ### Konvektiver Wärmeübergangskoeffizient
# 
# Für Innenräume ist der konvektive Wärmeübergangskoeffizient definiert als:
# 
# | Richtung des Wärmestroms | $\alpha_{c,i}$ in /m²K |
# |--------------------------|------------------------|
# | aufwärts                 | 5                      |
# | horizontal               | 2.5                    |
# | abwärts                  | 0.7                    |
# 
# Für Außenoberflächen ist $\alpha_{c,e}$ vorrangig abhängig von der Windgeschwindigkeit v in m/s:
# 
# $$ \alpha_{c,e} = 4 + 4 \cdot v $$
# 
# ### Radiativer Wärmeübergangskoeffizient
# 
# Der radiative Wärmeübergangskoeffizient $\alpha_{r}$ berechnet sich mittels des Emissionsgrades $\epsilon$, der
# Stefan-Boltzmann-Konstante $\sigma$ ($5.67 \cdot 10^{-8}$ W/m²K<sup>4</sup>), der mittleren Außenoberflächentemperatur
# $T_{se}$ und der mittleren Himmelstemperatur $T_{sky}$ in Kelvin.
# 
# $$ \alpha_{r} = \epsilon \cdot 4 \cdot \sigma \cdot \left(\frac{T_{se} + T_{sky}}{2}\right)^{3} $$ (alpha_r_eq)
# 
# Für übliche Verhältnisse kann der radiative Wärmeübergangskoeffizient jedoch angenähert werden zu:
# 
# $$ \alpha_{r} \approx 5 \cdot \epsilon $$
