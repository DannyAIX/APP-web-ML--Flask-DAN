/* ================================================================== */
/* static/js/script.js - JAVASCRIPT PERSONALIZADO */
/* ================================================================== */

// Esperar a que el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    
    // ====== Form Validation ======
    const form = document.getElementById('predictionForm');
    
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            } else {
                // Mostrar loading spinner
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn) {
                    submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Calculando...';
                    submitBtn.disabled = true;
                }
            }
            form.classList.add('was-validated');
        });
    }
    
    // ====== Auto-calculate BMI ======
    const weightInput = document.getElementById('weight');
    const heightInput = document.getElementById('height');
    
    if (weightInput && heightInput) {
        function calculateBMI() {
            const weight = parseFloat(weightInput.value);
            const height = parseFloat(heightInput.value);
            
            if (weight > 0 && height > 0) {
                const bmi = (weight / (height * height)).toFixed(2);
                
                // Mostrar BMI en algún lugar (opcional)
                console.log('BMI calculado:', bmi);
                
                // Agregar tooltip con BMI
                const bmiInfo = getBMICategory(bmi);
                if (weightInput.parentElement) {
                    let bmiLabel = weightInput.parentElement.querySelector('.bmi-info');
                    if (!bmiLabel) {
                        bmiLabel = document.createElement('small');
                        bmiLabel.className = 'bmi-info text-muted d-block mt-1';
                        weightInput.parentElement.appendChild(bmiLabel);
                    }
                    bmiLabel.innerHTML = `<i class="fas fa-info-circle"></i> Tu IMC: ${bmi} (${bmiInfo})`;
                }
            }
        }
        
        weightInput.addEventListener('input', calculateBMI);
        heightInput.addEventListener('input', calculateBMI);
    }
    
    // ====== Smooth Scrolling ======
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // ====== Tooltips Initialization (Bootstrap 5) ======
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // ====== Input Validation Feedback ======
    const inputs = document.querySelectorAll('.form-control, .form-select');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.value.trim() === '') {
                this.classList.add('is-invalid');
            } else {
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            }
        });
    });
    
    // ====== Nutrition Calculator Helper ======
    const proteinInput = document.getElementById('proteins');
    const carbsInput = document.getElementById('carbs');
    const fatsInput = document.getElementById('fats');
    const caloriesInput = document.getElementById('calories');
    
    if (proteinInput && carbsInput && fatsInput && caloriesInput) {
        function calculateTotalCalories() {
            const proteins = parseFloat(proteinInput.value) || 0;
            const carbs = parseFloat(carbsInput.value) || 0;
            const fats = parseFloat(fatsInput.value) || 0;
            
            // Calorías: Proteínas(4) + Carbos(4) + Grasas(9)
            const estimatedCalories = (proteins * 4) + (carbs * 4) + (fats * 9);
            
            // Actualizar campo de calorías si está vacío
            if (!caloriesInput.value || caloriesInput.value == 0) {
                caloriesInput.value = Math.round(estimatedCalories);
            }
            
            // Mostrar info
            let calorieInfo = caloriesInput.parentElement.querySelector('.calorie-info');
            if (!calorieInfo) {
                calorieInfo = document.createElement('small');
                calorieInfo.className = 'calorie-info text-muted d-block mt-1';
                caloriesInput.parentElement.appendChild(calorieInfo);
            }
            calorieInfo.innerHTML = `<i class="fas fa-calculator"></i> Calorías estimadas de macros: ${Math.round(estimatedCalories)}`;
        }
        
        proteinInput.addEventListener('input', calculateTotalCalories);
        carbsInput.addEventListener('input', calculateTotalCalories);
        fatsInput.addEventListener('input', calculateTotalCalories);
    }
    
    // ====== Animation on Scroll ======
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.feature-card, .card').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// ====== Helper Functions ======

function getBMICategory(bmi) {
    bmi = parseFloat(bmi);
    if (bmi < 18.5) return 'Bajo peso';
    if (bmi < 25) return 'Normal';
    if (bmi < 30) return 'Sobrepeso';
    return 'Obesidad';
}

// ====== API Call Example (for future use) ======
async function predictAPI(data) {
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('Error en la predicción');
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// ====== Print Results ======
function printResults() {
    window.print();
}

// ====== Share Results (Web Share API) ======
async function shareResults(fatPercentage, category) {
    if (navigator.share) {
        try {
            await navigator.share({
                title: 'Mi Porcentaje de Grasa Corporal',
                text: `Mi porcentaje de grasa corporal es ${fatPercentage}% (${category})`,
                url: window.location.href
            });
        } catch (error) {
            console.log('Error al compartir:', error);
        }
    } else {
        // Fallback: copiar al portapapeles
        const text = `Mi porcentaje de grasa corporal es ${fatPercentage}% (${category})`;
        navigator.clipboard.writeText(text);
        alert('¡Texto copiado al portapapeles!');
    }
}