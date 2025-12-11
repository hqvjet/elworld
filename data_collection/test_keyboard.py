"""
Script test keyboard input - kiểm tra phím nào được detect
"""
import keyboard
import time

KEYS_TO_LOG = [
    'f8', 'up', 'down', 'left', 'right', 'z', 'x',  # Basic cmd
    '1', '2', '3', '4', '5',                        # potion cmd
    'q', 'w', 'e', 'r', 't',                        # skill cmd
    'a', 's', 'd', 'c', 'f',                        # skill cmd
    'enter', 'ctrl', 'esc'
]

print("=" * 60)
print("🔍 TEST KEYBOARD INPUT - DETECT GHOST KEYS")
print("Bấm TỪNG PHÍM MỘT để test")
print("Đặc biệt test phím LEFT để xem có bị dính số 4 không")
print("Bấm F12 để thoát")
print("=" * 60)

last_pressed = set()

try:
    while True:
        pressed = set()
        for key in KEYS_TO_LOG:
            if keyboard.is_pressed(key):
                pressed.add(key)
        
        # Chỉ hiển thị khi có thay đổi
        if pressed != last_pressed:
            if pressed:
                combo = '+'.join(sorted(pressed))
                print(f"✅ Phím: {combo}")
                
                # Cảnh báo nếu detect cả left và 4 cùng lúc
                if 'left' in pressed and '4' in pressed:
                    print("   ⚠️  WARNING: Phát hiện LEFT + 4 cùng lúc! (Ghost key?)")
                if 'up' in pressed and '8' in pressed:
                    print("   ⚠️  WARNING: Phát hiện UP + 8 cùng lúc! (Ghost key?)")
                if 'right' in pressed and '6' in pressed:
                    print("   ⚠️  WARNING: Phát hiện RIGHT + 6 cùng lúc! (Ghost key?)")
                if 'down' in pressed and '2' in pressed:
                    print("   ⚠️  WARNING: Phát hiện DOWN + 2 cùng lúc! (Ghost key?)")
            else:
                print("   (không có phím nào)")
            
            last_pressed = pressed.copy()
        
        if keyboard.is_pressed('f12'):
            print("\n❌ Thoát...")
            break
        
        time.sleep(0.05)
        
except KeyboardInterrupt:
    print("\n❌ Dừng bởi Ctrl+C")

print("\n💡 Gợi ý:")
print("   - Nếu bấm LEFT nhưng hiện LEFT+4: Numpad đang bật hoặc ghost key")
print("   - Kiểm tra NumLock đã TẮT chưa")
print("   - Thử bấm phím mũi tên bên phải (không phải numpad)")
