"""
Ajout du cv.waitKey() manquant dans la boucle de main()
"""
import re

print("=" * 60)
print("CORRECTION DE LA BOUCLE MAIN() - cv.waitKey() MANQUANT")
print("=" * 60)

with open('app.py', 'r') as f:
    content = f.read()

# Trouver la boucle dans main() (pas dans generate_frames)
# Chercher "while True:" après "def main()" mais avant "def generate_frames"

# Trouver la position de main()
main_start = content.find('def main():')
if main_start == -1:
    print("❌ Fonction main() non trouvée")
    exit(1)

# Trouver generate_frames après main()
generate_start = content.find('def generate_frames():', main_start)
if generate_start == -1:
    print("❌ Fonction generate_frames() non trouvée après main()")
    exit(1)

# Extraire le contenu de main()
main_content = content[main_start:generate_start]

# Trouver le dernier "while True:" dans main()
while_positions = []
for match in re.finditer(r'while True:', main_content):
    while_positions.append(match.start())

if not while_positions:
    print("❌ Aucune boucle while True trouvée dans main()")
    exit(1)

# Prendre la dernière boucle dans main()
last_while = while_positions[-1]
while_start_in_main = last_while

print(f"📍 Dernière boucle dans main() à la position: {while_start_in_main}")

# Trouver la fin de cette boucle (rechercher la prochaine ligne non indentée)
main_lines = main_content.split('\n')
while_line_index = None

# Trouver l'index de la ligne du while
for i, line in enumerate(main_lines):
    if 'while True:' in line and i > while_line_index if while_line_index is not None else True:
        while_line_index = i
        break

if while_line_index is None:
    print("❌ Impossible de trouver la ligne du while")
    exit(1)

print(f"📍 Ligne du while dans main(): {while_line_index}")

# Trouver la fin de la boucle (ligne avec moins d'indentation)
indent_level = len(main_lines[while_line_index]) - len(main_lines[while_line_index].lstrip())
end_of_loop = None

for i in range(while_line_index + 1, len(main_lines)):
    line = main_lines[i]
    if line.strip():  # Ignorer les lignes vides
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= indent_level and not line.lstrip().startswith('#'):
            end_of_loop = i
            break

if end_of_loop is None:
    end_of_loop = len(main_lines) - 1

print(f"📍 Fin estimée de la boucle: ligne {end_of_loop}")

# Vérifier si cv.waitKey() est présent dans la boucle
has_waitkey = False
for i in range(while_line_index, end_of_loop):
    if 'cv.waitKey' in main_lines[i] or 'cv2.waitKey' in main_lines[i]:
        has_waitkey = True
        break

print(f"📍 cv.waitKey() présent: {'✅' if has_waitkey else '❌'}")

# Vérifier si cv.imshow() est présent
has_imshow = False
for i in range(while_line_index, end_of_loop):
    if 'cv.imshow' in main_lines[i] or 'cv2.imshow' in main_lines[i]:
        has_imshow = True
        imshow_line = i
        break

print(f"📍 cv.imshow() présent: {'✅' if has_imshow else '❌'}")

if has_imshow and not has_waitkey:
    print("\n🚨 PROBLÈME IDENTIFIÉ: cv.imshow() sans cv.waitKey() !")
    print("   La fenêtre s'ouvre mais ne se met pas à jour correctement")
    print("   et ne peut pas être fermée avec une touche.")
    
    # Trouver où insérer cv.waitKey() (après cv.imshow)
    insert_line = imshow_line + 1
    
    # Code à insérer
    waitkey_code = '''        # Wait for key press (1ms) and check for 'q' to quit
        key = cv.waitKey(1)
        if key == ord('q') or key == 27:  # 'q' or ESC
            print("Arrêt demandé par l'utilisateur")
            break'''
    
    # Insérer après cv.imshow()
    main_lines.insert(insert_line, waitkey_code)
    
    print(f"\n💡 CORRECTION: Ajout de cv.waitKey() après la ligne {imshow_line}")
    
    # Reconstruire main()
    corrected_main = '\n'.join(main_lines)
    
    # Remplacer dans le contenu original
    new_content = content[:main_start] + corrected_main + content[main_start + len(main_content):]
    
    # Sauvegarder
    with open('app.py.backup_waitkey', 'w') as f:
        f.write(content)
    
    with open('app.py.fixed_waitkey', 'w') as f:
        f.write(new_content)
    
    print("\n✅ FICHIERS CRÉÉS:")
    print("   • app.py.backup_waitkey - Copie originale")
    print("   • app.py.fixed_waitkey - Version corrigée")
    print("\n📋 POUR APPLIQUER:")
    print("   cp app.py.fixed_waitkey app.py")
    
    # Afficher le contexte corrigé
    print("\n📜 EXTRAIT CORRIGÉ (autour de cv.imshow):")
    for i in range(max(0, imshow_line-2), min(len(main_lines), imshow_line+6)):
        prefix = ">>> " if i == insert_line else "    "
        print(f"{prefix}{main_lines[i]}")
        
elif not has_imshow:
    print("\n⚠️ ATTENTION: Pas de cv.imshow() dans la boucle")
    print("   La fenêtre ne s'ouvrira pas visuellement")
    
else:
    print("\n✅ La boucle semble déjà avoir cv.waitKey()")

# Vérifier aussi hand_sign_letter
print("\n🔍 VÉRIFICATION DE hand_sign_letter:")
hand_sign_updates = []
for i in range(while_line_index, end_of_loop):
    if 'hand_sign_letter' in main_lines[i]:
        hand_sign_updates.append((i, main_lines[i].strip()))

if hand_sign_updates:
    print(f"✅ {len(hand_sign_updates)} mise(s) à jour de hand_sign_letter trouvée(s):")
    for line_num, code in hand_sign_updates:
        print(f"   Ligne {line_num}: {code}")
else:
    print("❌ Aucune mise à jour de hand_sign_letter dans la boucle!")
    print("   /sign retournera toujours des valeurs vides")

print("\n" + "=" * 60)
print("🎯 TEST APRÈS CORRECTION")
print("=" * 60)
print("""
Après avoir appliqué la correction:

1. Lancez le serveur:
   $ python app.py

2. Dans un autre terminal:
   $ curl http://127.0.0.1:5000/start
   # Attendez que la fenêtre s'ouvre
   $ curl http://127.0.0.1:5000/sign

3. Dans la fenêtre OpenCV:
   • Placez votre main devant la caméra
   • Appuyez sur 'q' pour quitter
   • Le signe détecté devrait s'afficher
""")
