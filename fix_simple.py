import re

with open('app.py', 'r') as f:
    content = f.read()

# Trouver où insérer la boucle simple
insert_point = content.find('finger_gesture_history = deque(maxlen=history_length)')
if insert_point == -1:
    print("Point d'insertion non trouvé")
    exit(1)

# Trouver la fin de la ligne
insert_point = content.find('\n', insert_point) + 1

# Boucle simple
simple_loop = '''
    # ########################################################################
    print("Démarrage de la boucle de reconnaissance...")
    
    while True:
        # Capture frame
        ret, image = cap.read()
        if not ret:
            print("Erreur caméra")
            break
            
        # Mirror display
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        
        # Convert to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Hand detection
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Preprocess
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Update global variable
                hand_sign_letter = keypoint_classifier_labels[hand_sign_id]
                send_hand_sign_letter()
                
                # Draw landmarks
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                # Display sign
                cv.putText(debug_image, f"Signe: {hand_sign_letter}", 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display FPS
        fps = cvFpsCalc.get()
        cv.putText(debug_image, f"FPS: {fps}", (10, 60), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show window
        cv.imshow('SignSpeak - Reconnaissance', debug_image)
        
        # Break on 'q' or ESC
        key = cv.waitKey(1)
        if key == ord('q') or key == 27:
            break
    
    cap.release()
    cv.destroyAllWindows()
    print("Reconnaissance terminée")
'''

# Créer le contenu corrigé
new_content = content[:insert_point] + simple_loop + content[insert_point:]

# Sauvegarder
with open('app.py.backup', 'w') as f:
    f.write(content)

with open('app.py.fixed_simple', 'w') as f:
    f.write(new_content)

print("✅ app.py.backup - Copie originale")
print("✅ app.py.fixed_simple - Version corrigée simple")
print("\nPour appliquer: cp app.py.fixed_simple app.py")
