CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT
);

CREATE TABLE products (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    brand VARCHAR(100),
    category_id INTEGER REFERENCES categories(id),
    price NUMERIC(12, 2) NOT NULL,
    currency VARCHAR(5) DEFAULT 'MAD',
    stock_quantity INTEGER DEFAULT 0,
    attributes JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Seed categories
INSERT INTO categories (name, description) VALUES
    ('Electronics', 'Smartphones, laptops, audio gear, and accessories'),
    ('Home & Kitchen', 'Appliances, cookware, and everyday home essentials'),
    ('Apparel & Footwear', 'Clothing, outerwear, and athletic or casual shoes'),
    ('Sports & Outdoors', 'Fitness equipment, cycling, and outdoor gear'),
    ('Beauty & Personal Care', 'Skincare, grooming, and daily wellness products');

-- Insert 20 products
INSERT INTO products (
    id, name, description, brand, category_id, price, currency, stock_quantity, attributes, is_active
) VALUES
    ('11111111-1111-4111-8111-111111111101', 'Galaxy A55 5G', 'Mid-range smartphone with AMOLED display and triple camera system', 'Samsung', 1, 4299.00, 'MAD', 35, '{"storage": "256GB", "ram": "8GB", "color": "Awesome Iceblue", "5g": true}', TRUE),
    ('11111111-1111-4111-8111-111111111102', 'iPhone 15', 'Smartphone with Dynamic Island, 48MP main camera, and USB-C', 'Apple', 1, 8999.00, 'MAD', 18, '{"storage": "128GB", "color": "Midnight", "5g": true}', TRUE),
    ('11111111-1111-4111-8111-111111111103', 'WH-1000XM5 Wireless Headphones', 'Industry-leading noise canceling over-ear headphones', 'Sony', 1, 3490.00, 'MAD', 22, '{"battery_hours": 30, "wireless": true, "color": "Black"}', TRUE),
    ('11111111-1111-4111-8111-111111111104', 'ThinkPad E14 Gen 5', 'Business laptop built for productivity and portability', 'Lenovo', 1, 7850.00, 'MAD', 12, '{"cpu": "Ryzen 7", "ram": "16GB", "ssd": "512GB", "screen_inch": 14}', TRUE),
    ('11111111-1111-4111-8111-111111111105', 'MX Master 3S Mouse', 'Ergonomic wireless mouse with quiet clicks and MagSpeed scroll', 'Logitech', 1, 1150.00, 'MAD', 40, '{"dpi": 8000, "wireless": true, "color": "Graphite"}', TRUE),

    ('22222222-2222-4222-8222-222222222201', 'Easy Fry Precision Air Fryer', 'Compact air fryer for crispy meals with little to no oil', 'Moulinex', 2, 1099.00, 'MAD', 28, '{"capacity_liters": 4.2, "power_watts": 1500, "color": "Black"}', TRUE),
    ('22222222-2222-4222-8222-222222222202', 'V60 Ceramic Coffee Dripper', 'Cone pour-over coffee dripper for clean extraction', 'Hario', 2, 240.00, 'MAD', 55, '{"size": "02", "material": "Ceramic", "color": "White"}', TRUE),
    ('22222222-2222-4222-8222-222222222203', 'Ingenico 5-Piece Cookware Set', 'Stackable non-stick pots and pans with detachable handle', 'Tefal', 2, 1390.00, 'MAD', 16, '{"pieces": 5, "induction_compatible": true, "coating": "Titanium"}', TRUE),
    ('22222222-2222-4222-8222-222222222204', 'Robot Vacuum E10', 'Slim automated vacuum cleaner for hard floors and rugs', 'Xiaomi', 2, 1899.00, 'MAD', 14, '{"suction_pa": 4000, "battery_mah": 2600, "color": "White"}', TRUE),
    ('22222222-2222-4222-8222-222222222205', 'Electric Gooseneck Kettle', 'Precision pouring kettle with temperature control', 'Fellow', 2, 1750.00, 'MAD', 9, '{"capacity_liters": 0.9, "power_watts": 1200, "color": "Matte Black"}', TRUE),

    ('33333333-3333-4333-8333-333333333301', 'Pegasus 41 Running Shoes', 'Responsive everyday road running shoes with dual Air Zoom units', 'Nike', 3, 1499.00, 'MAD', 30, '{"gender": "Unisex", "size": "42", "color": "Wolf Grey"}', TRUE),
    ('33333333-3333-4333-8333-333333333302', 'Classic Heavyweight Hoodie', 'Brushed cotton-blend fleece pullover hoodie', 'Adidas', 3, 649.00, 'MAD', 45, '{"material": "80% Cotton / 20% Polyester", "size": "L", "color": "Navy"}', TRUE),
    ('33333333-3333-4333-8333-333333333303', '511 Slim Fit Jeans', 'Everyday slim-cut denim jeans with slight stretch', 'Levis', 3, 790.00, 'MAD', 25, '{"waist": 32, "fit": "Slim", "color": "Dark Indigo"}', TRUE),
    ('33333333-3333-4333-8333-333333333304', 'Waterproof Rain Shell Jacket', 'Breathable windproof and waterproof outdoor jacket', 'Columbia', 3, 1190.00, 'MAD', 15, '{"waterproof": true, "size": "M", "color": "Olive"}', TRUE),
    ('33333333-3333-4333-8333-333333333305', 'Crew Performance Socks (3-Pack)', 'Moisture-wicking athletic crew socks with arch support', 'Under Armour', 3, 220.00, 'MAD', 60, '{"pack_size": 3, "size": "42-46", "color": "Black"}', TRUE),

    ('44444444-4444-4444-8444-444444444401', 'Pro Grip Yoga Mat 6mm', 'Non-slip cushioning mat for yoga, pilates, and stretching', 'Manduka', 4, 580.00, 'MAD', 32, '{"thickness_mm": 6, "material": "TPE", "color": "Slate"}', TRUE),
    ('44444444-4444-4444-8444-444444444402', 'Insulated Trail Bottle 1L', 'Double-wall stainless steel bottle keeping drinks cold 24h', 'Hydro Flask', 4, 420.00, 'MAD', 50, '{"capacity_liters": 1.0, "insulated": true, "color": "Cobalt"}', TRUE),
    ('44444444-4444-4444-8444-444444444403', 'Adjustable Dumbbell Pair (24kg)', 'Space-saving dial-adjustable dumbbells up to 24kg each', 'Bowflex', 4, 3890.00, 'MAD', 8, '{"max_weight_kg": 24, "increments": 15, "pair": true}', TRUE),

    ('55555555-5555-4555-8555-555555555501', 'Hydrating Facial Cleanser', 'Gentle daily cleanser with ceramides and hyaluronic acid', 'CeraVe', 5, 165.00, 'MAD', 75, '{"volume_ml": 236, "skin_type": "Normal to Dry"}', TRUE),
    ('55555555-5555-4555-8555-555555555502', 'Anthelios UVmune 400 Fluid SPF50+', 'Lightweight broad-spectrum sunscreen fluid for face', 'La Roche-Posay', 5, 215.00, 'MAD', 48, '{"spf": 50, "volume_ml": 50, "water_resistant": true}', FALSE);