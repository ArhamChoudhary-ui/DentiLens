"""Model labels and local checkpoint names for trained review modes."""
TOOTH_TYPES = ('central incisor', 'lateral incisor', 'canine', 'first premolar',
               'second premolar', 'first molar', 'second molar', 'third molar')
FDI_CLASSES = tuple(str(10 * quadrant + position)
                    for quadrant in range(1, 5) for position in range(1, 9))
CAVITY_CLASSES = ('possible cavity', 'normal appearance')
TRAINED_MODES = {
    'tooth-sam2': ('tooth-detector', 'Trained tooth outlines (experimental)', ('tooth',)),
    'tooth-types': ('tooth-types', 'Tooth type names (experimental)', TOOTH_TYPES),
    'tooth-fdi': ('tooth-fdi', 'Tooth numbers (experimental FDI)', FDI_CLASSES),
    'cavity-review': ('cavity-review', 'Possible cavity flags (experimental)', CAVITY_CLASSES),
}


def fdi_name(code):
    sides = ('upper right', 'upper left', 'lower left', 'lower right')
    return f'{sides[int(code[0]) - 1]} {TOOTH_TYPES[int(code[1]) - 1]}'
