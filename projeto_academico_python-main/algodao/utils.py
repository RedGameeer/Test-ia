def format_number(value, prefix=''):
    """
    Formata o número com prefixo e adequação de unidades, como mil e milhões.
    """
    for unit in ['', 'mil', 'milhões']:
        if value < 1000:
            return f'{prefix} {value:,.2f} {unit}'
        value /= 1000
    return f'{prefix} {value:,.2f} milhões'

