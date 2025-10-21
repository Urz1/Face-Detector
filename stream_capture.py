import cv2
import time
import numpy as np
import requests
import os

def save_on_movement(stream_url, output_path, screenshot_path, record_time=10, min_area=1500):
    def open_stream(url):
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Failed to open stream: {url}")
            return None
        return cap

    cap = open_stream(stream_url)
    if not cap:
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    target_fps = 25.0  # Fixed frame rate for recording
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    if not ret:
        print("Failed to read initial frames")
        cap.release()
        return

    window_name = "Live Cam"
    cv2.namedWindow(window_name)
    print("Monitoring for movement... Press ESC to exit.")
    
    server_url = "http://localhost:5000/process_image"
    api_key = "5gl5TTvpUaX3J9K-muC3pSiTfGHy8S_Vn3ruNk8Vnqw"

    telegram_bot_token = "7542006617:AAHjxV0IUEvDyT2maDWDgtVsVlFvsCUbiM8"
    telegram_chat_id = "928614686"
    telegram_photo_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendPhoto"
    telegram_video_url = f"https://api.telegram.org/bot{telegram_bot_token}/sendVideo"

    try:
        while True:
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            movement = any(cv2.contourArea(contour) >= min_area for contour in contours)

            if movement:
                print("Movement detected! Capturing screenshot and recording for 10 seconds...")
                time.sleep(3)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(screenshot_path, frame)
                    print(f"Screenshot saved to {screenshot_path}")

                    if os.path.exists(screenshot_path):
                        face_result = "Face detection failed"
                        try:
                            with open(screenshot_path, 'rb') as img_file:
                                headers = {'X-API-Key': api_key}
                                response = requests.post(server_url, data=img_file, headers=headers)
                            
                            if response.status_code == 200:
                                result = response.json()
                                face_result = result['message']
                                print(f"Face detection result: {face_result}")
                            else:
                                print(f"Server error: {response.status_code} - {response.text}")
                        except requests.exceptions.RequestException as e:
                            print(f"Failed to send image to server: {e}")
                        except ValueError as e:
                            print(f"Invalid server response: {e}")

                        try:
                            with open(screenshot_path, 'rb') as img_file:
                                files = {'photo': img_file}
                                data = {
                                    'chat_id': telegram_chat_id,
                                    'caption': f"Motion detected at {time.strftime('%Y-%m-%d %H:%M:%S')}\nFace detection: {face_result}"
                                }
                                response = requests.post(telegram_photo_url, files=files, data=data)
                            
                            if response.status_code == 200:
                                print("Screenshot sent to Telegram bot successfully")
                            else:
                                print(f"Telegram API error (photo): {response.status_code} - {response.text}")
                        except requests.exceptions.RequestException as e:
                            print(f"Failed to send image to Telegram: {e}")
                        except Exception as e:
                            print(f"Error sending to Telegram (photo): {e}")
                    else:
                        print(f"Screenshot file not found: {screenshot_path}")

                else:
                    print("Failed to capture screenshot frame.")

                print("Starting video recording...")
                out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
                total_frames = int(target_fps * record_time)
                frame_interval = 1.0 / target_fps
                frames_written = 0

                try:
                    start_time = time.time()
                    for i in range(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            print(f"Frame {i+1}/{total_frames}: Stream lost, attempting to reconnect...")
                            cap.release()
                            cap = open_stream(stream_url)
                            if not cap:
                                print("Reconnection failed, aborting recording")
                                break
                            ret, frame = cap.read()
                            if not ret:
                                print("Failed to read frame after reconnection")
                                break

                        out.write(frame)
                        frames_written += 1
                        if frames_written % 50 == 0:
                            print(f"Recorded {frames_written}/{total_frames} frames")

                        elapsed = time.time() - start_time
                        expected_time = (i + 1) * frame_interval
                        if elapsed < expected_time:
                            time.sleep(expected_time - elapsed)

                        if cv2.waitKey(1) == 27:
                            print("ESC pressed, stopping recording")
                            break
                except Exception as e:
                    print(f"Recording error: {e}")
                finally:
                    out.release()
                    print(f"Recording finished. Wrote {frames_written} frames")

                    # Send video to Telegram
                    if os.path.exists(output_path):
                        video_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
                        if video_size <= 50:
                            try:
                                with open(output_path, 'rb') as video_file:
                                    files = {'video': video_file}
                                    data = {
                                        'chat_id': telegram_chat_id,
                                        'caption': f"Motion detected at {time.strftime('%Y-%m-%d %H:%M:%S')}\nFace detection: {face_result}"
                                    }
                                    print("Sending video to Telegram...")
                                    response = requests.post(telegram_video_url, files=files, data=data)
                                
                                if response.status_code == 200:
                                    print("Video sent to Telegram bot successfully")
                                else:
                                    print(f"Telegram API error (video): {response.status_code} - {response.text}")
                            except requests.exceptions.RequestException as e:
                                print(f"Failed to send video to Telegram: {e}")
                            except Exception as e:
                                print(f"Error sending to Telegram (video): {e}")
                        else:
                            print(f"Video too large ({video_size:.2f} MB). Telegram limit is 50 MB. Reduce resolution or record_time.")
                    else:
                        print(f"Video file not found: {output_path}")

                print("Waiting for new movement...")

                while movement:
                    ret, frame1 = cap.read()
                    if not ret:
                        print("Stream lost in movement skip, reconnecting...")
                        cap.release()
                        cap = open_stream(stream_url)
                        if not cap:
                            print("Reconnection failed, exiting")
                            return
                        ret, frame1 = cap.read()
                        if not ret:
                            print("Failed to read frame after reconnection")
                            return
                    ret, frame2 = cap.read()
                    if not ret:
                        print("Stream lost in movement skip, reconnecting...")
                        cap.release()
                        cap = open_stream(stream_url)
                        if not cap:
                            print("Reconnection failed, exiting")
                            return
                        ret, frame2 = cap.read()
                        if not ret:
                            print("Failed to read frame after reconnection")
                            return

                    diff = cv2.absdiff(frame1, frame2)
                    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0)
                    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
                    dilated = cv2.dilate(thresh, None, iterations=2)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    movement = any(cv2.contourArea(contour) >= min_area for contour in contours)
                    cv2.imshow(window_name, frame1)
                    if cv2.waitKey(1) == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        return
            else:
                frame1 = frame2
                ret, frame2 = cap.read()
                if not ret:
                    print("Stream lost, attempting to reconnect...")
                    cap.release()
                    cap = open_stream(stream_url)
                    if not cap:
                        print("Reconnection failed, exiting")
                        return
                    ret, frame2 = cap.read()
                    if not ret:
                        print("Failed to read frame after reconnection")
                        return
                cv2.imshow(window_name, frame1)
                if cv2.waitKey(10) == 27:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")

if __name__ == "__main__":
    stream_url = "http://192.168.137.166/mjpeg/1"
    output_path = "esp32_stream.avi"
    screenshot_path = "esp32_stream_screenshot.jpg"
    save_on_movement(stream_url, output_path, screenshot_path)