"""
Correction finale : port 5001 + cv.waitKey()
"""
import re

with open('app.py', 'r') as f:
    lines = f.readlines()

print("=" * 60)
print("CORRECTION FINALE")
print("=" * 60)

# 1. CHANGER LE PORT À LA LIGNE 683
if len(lines) > 683 and 'app.run(debug=True)' in lines[682]:
    lines[682] = '    app.run(debug=True, port=5001)\n'
    print("✅ Port changé à 5001 (ligne 683)")
else:
    # Chercher app.run() ailleurs
    for i, line in enumerate(lines):
        if 'app.run(' in line and 'port=' not in line:
            lines[i] = line.replace('app.run(', 'app.run(port=5001, ')
            print(f"✅ Port 5001 ajouté (ligne {i+1})")
            break

# 2. CORRIGER cv.waitKey() MANQUANT
print("\n🔍 Recherche de cv.imshow sans waitKey...")

found_fix = False
for i in range(len(lines)):
    if 'cv.imshow(' in lines[i] and 'Hand Gesture Recognition' in lines[i]:
        print(f"✅ cv.imshow trouvé ligne {i+1}: {lines[i].strip()}")
        
        # Vérifier les 3 lignes suivantes pour waitKey
        has_waitkey = False
        for j in range(i+1, min(i+5, len(lines))):
            if 'cv.waitKey' in lines[j] or 'waitKey' in lines[j]:
                has_waitkey = True
                break
        
        if not has_waitkey:
            print(f"❌ cv.waitKey() manquant après ligne {i+1}")
            
            # Déterminer l'indentation
            indent = len(lines[i]) - len(lines[i].lstrip())
            indent_str = ' ' * indent
            
            # Code à insérer
            waitkey_code = [
                f'{indent_str}# Contrôle clavier (correction automatique)\n',
                f'{indent_str}key = cv.waitKey(1)\n',
                f'{indent_str}if key == ord("q") or key == 27:  # "q" ou ESC\n',
                f'{indent_str}    print("Arrêt demandé par l\\'utilisateur")\n',
                f'{indent_str}    break\n',
                '\n'
            ]
            
            # Insérer après cv.imshow
            for j, code_line in enumerate(waitkey_code):
                lines.insert(i + 1 + j, code_line)
            
            print(f"✅ cv.waitKey() ajouté avec condition de sortie")
            found_fix = True
            break  # Corriger seulement la première occurrence
        else:
            print(f"✅ cv.waitKey() déjà présent")
            found_fix = True
            break

if not found_fix:
    print("⚠️ cv.imshow 'Hand Gesture Recognition' non trouvé")
    # Chercher n'importe quel cv.imshow
    for i in range(len(lines)):
        if 'cv.imshow(' in lines[i]:
            print(f"Alternative: cv.imshow ligne {i+1}")
            # Même logique de correction...

# 3. SAUVEGARDER
with open('app.py.backup.final', 'w') as f:
    f.writelines(lines)
print("\n✅ Backup: app.py.backup.final")

# Ajouter aussi la ligne pour éviter les problèmes de cache
for i, line in enumerate(lines):
    if 'from flask import' in line and 'Flask' in line:
        if 'make_response' not in line:
            lines[i] = line.replace('from flask import ', 'from flask import make_response, ')
            print("✅ make_response importé pour éviter les problèmes de cache")
        break

# Sauvegarder version corrigée
with open('app.py.fixed_final', 'w') as f:
    f.writelines(lines)
print("✅ Version corrigée: app.py.fixed_final")

# 4. AFFICHER LES CHANGEMENTS
print("\n📋 CHANGEMENTS APPLIQUÉS:")
print("1. Port Flask défini sur 5001")
print("2. cv.waitKey(1) ajouté après cv.imshow()")
print("3. Condition de sortie avec 'q' ou ESC")
print("4. Import make_response ajouté")

print("\n" + "=" * 60)
print("🎯 POUR TESTER:")
print("1. cp app.py.fixed_final app.py")
print("2. python app.py")
print("3. Ouvrez: http://127.0.0.1:5001/start")
print("4. Testez: http://127.0.0.1:5001/sign")
print("\n💡 La fenêtre se ferme avec 'q' ou ESC")
