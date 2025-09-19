import math
import re

def validate_integer(number_str: str) -> bool:
    """Validate that string represents a positive integer."""
    if not isinstance(number_str, str):
        return False
    
    # Check if string contains only digits
    if not re.match(r'^\d+$', number_str):
        return False
    
    # Check for leading zeros (except single "0")
    if len(number_str) > 1 and number_str[0] == '0':
        return False
        
    return True

def calculate_bit_length(number_str: str) -> int:
    """Calculate bit length of a number given as string."""
    if not validate_integer(number_str):
        raise ValueError(f"Invalid number format: {number_str}")
    
    number = int(number_str)
    if number == 0:
        return 1
    
    return number.bit_length()

def calculate_digit_length(number_str: str) -> int:
    """Calculate decimal digit length of a number given as string."""
    if not validate_integer(number_str):
        raise ValueError(f"Invalid number format: {number_str}")
    
    return len(number_str)

def gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor using Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a

def is_trivial_factor(factor: str, composite: str) -> bool:
    """Check if factor is trivial (1 or the number itself)."""
    return factor == "1" or factor == composite

def verify_complete_factorization(composite: str, factors: list[str]) -> bool:
    """
    Verify that the product of factors equals the composite.
    Extracted from FactorService for better separation of concerns.
    """
    if not factors:
        return False
    
    try:
        # Calculate product of all factors
        product = 1
        for factor in factors:
            if not validate_integer(factor):
                return False
            product *= int(factor)
        
        return str(product) == composite
    except (ValueError, OverflowError):
        return False