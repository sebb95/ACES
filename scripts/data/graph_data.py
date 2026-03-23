import matplotlib.pyplot as plt

# Data
arts = ['Sei', 'Torsk', 'Lyr', 'Uer', 'Hyse', 'Brosme', 'Kveite', 'Lange', 'Breiflabb',"Bifangst" , 'Ukjent']
antall = [1325, 291, 196, 104, 228, 105, 103, 79, 26, 47, 140]

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(arts, antall, color='skyblue')

# Marker hovedarter
bars[0].set_color('navy') # Sei
bars[1].set_color('navy') # Torsk

plt.title('Artsfordeling i treningsdatasett (N=2644)')
plt.ylabel('Antall individer')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('artsfordeling.png', dpi=300) # Lagrer grafen som bilde til PPT
plt.show()